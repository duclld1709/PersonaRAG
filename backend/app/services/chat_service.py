"""Chat orchestration service for the RAG workflow."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.config import Settings
from app.domain.interfaces.llm_client import LLMClient
from app.domain.interfaces.vector_store import VectorStore
from app.domain.models.chat import ChatMessage, Citation, Conversation, MessageRole
from app.services.conversation_service import ConversationService

try:
    from langsmith.run_helpers import traceable
except ImportError:  # pragma: no cover - optional dependency
    def traceable(*dargs: Any, **dkwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return decorator


class QueryVariations(BaseModel):
    """Structured response for multi-query generation."""

    queries: list[str]


@dataclass
class ChatGenerationResult:
    """Container for generated assistant answer and supporting citations."""

    answer: str
    citations: list[Citation]


class ChatService:
    """High-level service that handles retrieval and response generation."""

    def __init__(
        self,
        *,
        settings: Settings,
        conversation_service: ConversationService,
        vector_store: VectorStore,
        llm_client: LLMClient,
    ) -> None:
        self._settings = settings
        self._conversation_service = conversation_service
        self._vector_store = vector_store
        self._llm = llm_client

    def update_llm_client(self, llm_client: LLMClient) -> None:
        """Swap the active LLM client without rebuilding other dependencies."""
        self._llm = llm_client

    @traceable(name="chat_service.stream_answer", run_type="chain")
    def stream_answer(
        self,
        conversation_id: UUID,
        question: str,
        *,
        run_tree: Any | None = None,
    ) -> Iterable[dict[str, Any]]:
        """Generate an answer for the given question and yield streaming events."""

        conversation = self._conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        history_before_question = list(conversation.messages)
        search_question = self._rewrite_question(history_before_question, question)

        self._conversation_service.add_user_message(conversation_id, question)

        retrieved_lists = self._multi_query_retrieval(search_question)
        fused_documents = self._reciprocal_rank_fusion(retrieved_lists)
        self._log_retrieval_run(run_tree, search_question, fused_documents)
        citations = self._build_citations(fused_documents)

        messages = self._build_prompt(history_before_question, question, fused_documents)

        final_answer: list[str] = []
        for token in self._llm.stream(messages):
            final_answer.append(token)
            yield {"delta": token}

        combined_answer = "".join(final_answer).strip()
        if combined_answer:
            self._conversation_service.add_assistant_message(
                conversation_id,
                combined_answer,
                citations=citations if citations else None,
            )
        if citations:
            citation_payload = [asdict(citation) for citation in citations]
            self._log_citations(run_tree, citations)
            yield {"citations": citation_payload}

    @traceable(name="chat_service.generate_answer", run_type="chain")
    def generate_answer(
        self,
        conversation_id: UUID,
        question: str,
        *,
        run_tree: Any | None = None,
    ) -> ChatGenerationResult:
        """Generate an answer without streaming."""

        conversation = self._conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        history_before_question = list(conversation.messages)
        search_question = self._rewrite_question(history_before_question, question)

        self._conversation_service.add_user_message(conversation_id, question)

        retrieved_lists = self._multi_query_retrieval(search_question)
        fused_documents = self._reciprocal_rank_fusion(retrieved_lists)
        self._log_retrieval_run(run_tree, search_question, fused_documents)
        citations = self._build_citations(fused_documents)

        messages = self._build_prompt(history_before_question, question, fused_documents)
        completion = self._llm.invoke(messages)
        answer = completion.content.strip()

        if answer:
            self._conversation_service.add_assistant_message(
                conversation_id,
                answer,
                citations=citations if citations else None,
            )

        result = ChatGenerationResult(answer=answer, citations=citations)
        self._log_citations(run_tree, citations)
        return result

    def _rewrite_question(self, history: Sequence[ChatMessage], new_question: str) -> str:
        if not history:
            return new_question

        system_prompt = SystemMessage(
            content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.",
        )

        lc_messages: list[BaseMessage] = [system_prompt] + self._to_langchain_messages(history)
        lc_messages.append(HumanMessage(content=f"New question: {new_question}"))

        response = self._llm.invoke(lc_messages)
        return response.content.strip()

    def _multi_query_retrieval(self, question: str) -> list[list[Document]]:
        structured_llm = self._llm.with_structured_output(QueryVariations)
        prompt = (
            f"Generate {self._settings.query_variations} different variations of this query "
            f"that would help retrieve relevant documents.\n\nOriginal query: {question}\n"
            "Return alternative queries that rephrase or approach the same problem."
        )
        response: QueryVariations = structured_llm.invoke(prompt)
        queries = response.queries or []

        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in queries:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            fingerprint = cleaned.lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            normalized.append(cleaned)
            if len(normalized) >= self._settings.query_variations:
                break

        request_question = question.strip()
        if not normalized:
            normalized = [request_question]
        elif request_question.lower() not in seen:
            normalized.insert(0, request_question)

        normalized = normalized[: self._settings.query_variations]

        retriever = self._vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self._settings.retrieval_k,
                "fetch_k": self._settings.retrieval_fetch_k,
                "lambda_mult": self._settings.retrieval_lambda,
            },
        )

        results: list[list[Document]] = []
        for query in normalized:
            docs = retriever.invoke(query)
            results.append(docs)
        return results

    def _reciprocal_rank_fusion(self, chunk_lists: list[list[Document]]) -> list[Document]:
        if not chunk_lists:
            return []

        rrf_scores: defaultdict[str, float] = defaultdict(float)
        chunk_lookup: dict[str, Document] = {}

        for chunks in chunk_lists:
            for position, chunk in enumerate(chunks, start=1):
                chunk_id = self._document_identifier(chunk)
                chunk_lookup[chunk_id] = chunk
                rrf_scores[chunk_id] += 1 / (self._settings.rrf_k + position)

        sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [chunk_lookup[key] for key, _ in sorted_chunks][:5]

    def _build_prompt(self, history: Sequence[ChatMessage], question: str, documents: Sequence[Document]) -> list[BaseMessage]:
        system_message = SystemMessage(content="You are a helpful assistant.")
        history_messages = self._to_langchain_messages(history)

        context_sections = "\n".join(f"- {doc.page_content}" for doc in documents)
        combined_input = (
            f"Based on the following documents, please answer this question: {question}\n\n"
            f"Documents:\n{context_sections}\n\n"
            'Please provide a clear, helpful answer using only the information from these documents. '
            'If you cannot find the answer in the documents, say "I do not have enough information to answer that question based on the provided documents."'
        )

        messages: list[BaseMessage] = [system_message] + history_messages
        messages.append(HumanMessage(content=combined_input))
        return messages

    def _build_citations(self, documents: Sequence[Document]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int | None, str]] = set()

        for document in documents:
            metadata = document.metadata or {}
            source = str(metadata.get("source") or metadata.get("file_path") or "unknown")
            filename = Path(source).name if source and source != "unknown" else "unknown"

            raw_page = metadata.get("page", metadata.get("page_number"))
            page: int | None
            if isinstance(raw_page, int):
                page = raw_page
            elif isinstance(raw_page, str) and raw_page.isdigit():
                page = int(raw_page)
            else:
                page = None

            if source.lower().endswith(".txt"):
                page = None

            content = document.page_content.strip()
            if not content:
                continue

            identity = (source, page, content)
            if identity in seen:
                continue
            seen.add(identity)

            citations.append(
                Citation(
                    source=source,
                    filename=filename,
                    page=page,
                    content=content,
                )
            )

        return citations

    def _to_langchain_messages(self, messages: Sequence[ChatMessage]) -> list[BaseMessage]:
        lc_messages: list[BaseMessage] = []
        for message in messages:
            if message.role == MessageRole.USER:
                lc_messages.append(HumanMessage(content=message.content))
            elif message.role == MessageRole.ASSISTANT:
                lc_messages.append(AIMessage(content=message.content))
            elif message.role == MessageRole.SYSTEM:
                lc_messages.append(SystemMessage(content=message.content))
        return lc_messages

    def _document_identifier(self, document: Document) -> str:
        source = document.metadata.get("source", "unknown")
        return f"{source}:{hash(document.page_content)}"

    def _log_retrieval_run(
        self,
        run_tree: Any | None,
        question: str,
        documents: Sequence[Document],
    ) -> None:
        if not run_tree or not hasattr(run_tree, "create_child"):
            return
        try:
            child = run_tree.create_child(
                name="retrieval",
                run_type="retriever",
                inputs={"question": question},
            )
            child.end(
                outputs={
                    "documents": self._snapshot_documents(documents),
                }
            )
        except Exception:
            pass

    def _log_citations(self, run_tree: Any | None, citations: Sequence[Citation]) -> None:
        if not run_tree or not citations or not hasattr(run_tree, "create_child"):
            return
        try:
            child = run_tree.create_child(name="citations", run_type="chain", inputs={})
            child.end(outputs={"citations": [asdict(item) for item in citations]})
        except Exception:
            pass

    def _snapshot_documents(self, documents: Sequence[Document]) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for document in documents:
            metadata = document.metadata or {}
            snapshot.append(
                {
                    "document_id": metadata.get("document_id"),
                    "source": metadata.get("source"),
                    "filename": metadata.get("filename"),
                    "chunk_index": metadata.get("chunk_index"),
                    "page": metadata.get("page") or metadata.get("page_number"),
                    "preview": document.page_content[:200],
                }
            )
        return snapshot
