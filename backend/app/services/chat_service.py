"""Chat orchestration service for the research-assistant RAG workflow.

Pipeline:
    route(question) →
        if refuse:    return refusal
        if chitchat:  answer without retrieval
        if retrieve:  cache lookup → hybrid retrieve → rerank → generate → cache store
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.config import Settings
from app.core.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    ChatMode,
    MODE_INSTRUCTIONS,
    QUERY_EXPANSION_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
    RESEARCH_ASSISTANT_SYSTEM_PROMPT,
    build_user_prompt,
)
from app.domain.interfaces.embedding_client import EmbeddingClient
from app.domain.interfaces.vector_store import VectorStore
from app.domain.models.chat import ChatMessage, Citation, Conversation, MessageRole, TokenUsage
from app.infrastructure.llm.openai_chat import OpenAIChatClient, OpenAIUtilityClient
from app.infrastructure.moderation.openai_moderation import OpenAIModerationClient
from app.infrastructure.persistence.sqlite_usage_repository import SqliteUsageRepository, UsageEvent
from app.retrieval import (
    HybridRetriever,
    LLMReranker,
    QueryIntent,
    QueryRouter,
    RetrievalHit,
    SemanticResponseCache,
)
from app.services.conversation_service import ConversationService

logger = getLogger(__name__)

try:
    from langsmith.run_helpers import traceable
except ImportError:  # pragma: no cover

    def traceable(*dargs: Any, **dkwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return decorator


class _QueryVariations(BaseModel):
    """Structured response from the expansion model."""

    queries: list[str]


@dataclass
class ChatGenerationResult:
    """Container for a generated assistant answer and supporting citations."""

    answer: str
    citations: list[Citation]
    usage: TokenUsage
    latency_ms: int
    mode: str
    cache_hit: bool = False
    refused: bool = False


class ChatService:
    """High-level service orchestrating the RAG pipeline end to end."""

    def __init__(
        self,
        *,
        settings: Settings,
        conversation_service: ConversationService,
        vector_store: VectorStore,
        chat_client: OpenAIChatClient,
        utility_client: OpenAIUtilityClient,
        embedding_client: EmbeddingClient,
        moderation_client: OpenAIModerationClient,
        retriever: HybridRetriever,
        reranker: LLMReranker,
        router: QueryRouter,
        cache: SemanticResponseCache,
        usage_repository: SqliteUsageRepository,
    ) -> None:
        self._settings = settings
        self._conversation_service = conversation_service
        self._vector_store = vector_store
        self._chat = chat_client
        self._utility = utility_client
        self._embeddings = embedding_client
        self._moderation = moderation_client
        self._retriever = retriever
        self._reranker = reranker
        self._router = router
        self._cache = cache
        self._usage = usage_repository

    # ------------------------------------------------------------------
    # Streaming entrypoint
    # ------------------------------------------------------------------
    @traceable(name="chat_service.stream_answer", run_type="chain")
    def stream_answer(
        self,
        conversation_id: UUID,
        question: str,
        *,
        mode: ChatMode = ChatMode.ANSWER,
        run_tree: Any | None = None,
    ) -> Iterable[dict[str, Any]]:
        start = time.perf_counter()

        conversation = self._conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        history_before = list(conversation.messages)

        # 1. Moderation — student safety & abuse prevention.
        if self._settings.enable_moderation:
            moderation = self._moderation.check(question)
            if moderation.flagged:
                yield from self._finalize_refusal(
                    conversation_id,
                    question,
                    "Your question was flagged by content moderation — please rephrase.",
                    mode=mode,
                    start=start,
                )
                return

        # 2. Intent routing.
        yield {"status": "routing"}
        intent = self._router.classify(question)

        if intent == QueryIntent.REFUSE:
            yield from self._finalize_refusal(
                conversation_id, question, REFUSAL_MESSAGE, mode=mode, start=start
            )
            return

        if intent == QueryIntent.CHITCHAT:
            yield from self._handle_chitchat(conversation_id, history_before, question, mode, start)
            return

        # 3. Semantic cache lookup.
        cached = self._cache.lookup(question)
        if cached:
            yield {"status": "cache_hit"}
            self._conversation_service.add_user_message(conversation_id, question)
            latency_ms = int((time.perf_counter() - start) * 1000)
            usage = TokenUsage()
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=cached.answer,
                citations=list(cached.citations) or None,
                usage=usage,
                latency_ms=latency_ms,
                mode=mode.value,
            )
            self._conversation_service.append_assistant(conversation_id, assistant_msg)
            yield {"delta": cached.answer}
            if cached.citations:
                yield {"citations": [asdict(c) for c in cached.citations]}
            yield {
                "usage": asdict(usage),
                "latency_ms": latency_ms,
                "cache_hit": True,
                "mode": mode.value,
            }
            self._usage.record(
                UsageEvent(
                    conversation_id=str(conversation_id),
                    mode=mode.value,
                    model=self._chat.model_name,
                    usage=usage,
                    latency_ms=latency_ms,
                    cache_hit=True,
                )
            )
            return

        # 4. Query rewrite + expansion for retrieval.
        yield {"status": "retrieving"}
        search_query = self._rewrite_question(history_before, question)
        expansions = self._expand_query(search_query)

        # 5. Hybrid search + rerank.
        hits = self._retriever.retrieve(
            [search_query, *expansions],
            k=self._settings.retrieval_k,
            fetch_k=self._settings.retrieval_fetch_k,
        )
        yield {"status": "reranking"}
        top_hits = self._reranker.rerank(
            question, hits, top_n=self._settings.rerank_top_n
        )
        citations = self._build_citations(top_hits)
        self._log_retrieval_run(run_tree, search_query, top_hits)

        # 6. Generate streamed answer.
        yield {"status": "generating"}
        self._conversation_service.add_user_message(conversation_id, question)
        messages = self._build_prompt(history_before, question, top_hits, mode)
        accumulated: list[str] = []
        final_usage: TokenUsage | None = None

        for delta, usage in self._chat.stream_with_usage(messages):
            if delta:
                accumulated.append(delta)
                yield {"delta": delta}
            if usage is not None:
                final_usage = usage

        answer = "".join(accumulated).strip()
        usage = final_usage or TokenUsage()
        latency_ms = int((time.perf_counter() - start) * 1000)

        if answer:
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=answer,
                citations=citations or None,
                usage=usage,
                latency_ms=latency_ms,
                mode=mode.value,
            )
            self._conversation_service.append_assistant(conversation_id, assistant_msg)
            self._cache.store(question, answer, citations)

        if citations:
            yield {"citations": [asdict(c) for c in citations]}
        yield {
            "usage": asdict(usage),
            "latency_ms": latency_ms,
            "cache_hit": False,
            "mode": mode.value,
        }
        self._usage.record(
            UsageEvent(
                conversation_id=str(conversation_id),
                mode=mode.value,
                model=self._chat.model_name,
                usage=usage,
                latency_ms=latency_ms,
                cache_hit=False,
            )
        )

    # ------------------------------------------------------------------
    # Non-streaming entrypoint (used by eval harness + simple clients)
    # ------------------------------------------------------------------
    @traceable(name="chat_service.generate_answer", run_type="chain")
    def generate_answer(
        self,
        conversation_id: UUID,
        question: str,
        *,
        mode: ChatMode = ChatMode.ANSWER,
        run_tree: Any | None = None,
    ) -> ChatGenerationResult:
        start = time.perf_counter()

        conversation = self._conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        if self._settings.enable_moderation and self._moderation.check(question).flagged:
            return self._refusal_result(
                conversation_id,
                question,
                "Your question was flagged by content moderation — please rephrase.",
                mode=mode,
                start=start,
            )

        intent = self._router.classify(question)
        if intent == QueryIntent.REFUSE:
            return self._refusal_result(conversation_id, question, REFUSAL_MESSAGE, mode=mode, start=start)

        if intent == QueryIntent.CHITCHAT:
            return self._chitchat_result(conversation_id, conversation.messages, question, mode, start)

        cached = self._cache.lookup(question)
        if cached:
            self._conversation_service.add_user_message(conversation_id, question)
            latency_ms = int((time.perf_counter() - start) * 1000)
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=cached.answer,
                citations=list(cached.citations) or None,
                usage=TokenUsage(),
                latency_ms=latency_ms,
                mode=mode.value,
            )
            self._conversation_service.append_assistant(conversation_id, assistant_msg)
            return ChatGenerationResult(
                answer=cached.answer,
                citations=list(cached.citations),
                usage=TokenUsage(),
                latency_ms=latency_ms,
                mode=mode.value,
                cache_hit=True,
            )

        history = list(conversation.messages)
        search_query = self._rewrite_question(history, question)
        expansions = self._expand_query(search_query)
        hits = self._retriever.retrieve(
            [search_query, *expansions],
            k=self._settings.retrieval_k,
            fetch_k=self._settings.retrieval_fetch_k,
        )
        top_hits = self._reranker.rerank(question, hits, top_n=self._settings.rerank_top_n)
        citations = self._build_citations(top_hits)
        self._log_retrieval_run(run_tree, search_query, top_hits)

        self._conversation_service.add_user_message(conversation_id, question)
        messages = self._build_prompt(history, question, top_hits, mode)
        response = self._chat.invoke(messages)
        answer = (response.content if isinstance(response, AIMessage) else str(response.content)).strip()
        usage = _usage_from_response(response) or TokenUsage()
        latency_ms = int((time.perf_counter() - start) * 1000)

        if answer:
            self._conversation_service.append_assistant(
                conversation_id,
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=answer,
                    citations=citations or None,
                    usage=usage,
                    latency_ms=latency_ms,
                    mode=mode.value,
                ),
            )
            self._cache.store(question, answer, citations)

        self._usage.record(
            UsageEvent(
                conversation_id=str(conversation_id),
                mode=mode.value,
                model=self._chat.model_name,
                usage=usage,
                latency_ms=latency_ms,
                cache_hit=False,
            )
        )
        return ChatGenerationResult(
            answer=answer,
            citations=citations,
            usage=usage,
            latency_ms=latency_ms,
            mode=mode.value,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        history: Sequence[ChatMessage],
        question: str,
        hits: Sequence[RetrievalHit],
        mode: ChatMode,
    ) -> list[BaseMessage]:
        system = SystemMessage(content=RESEARCH_ASSISTANT_SYSTEM_PROMPT)
        trimmed = self._trim_history(history)
        history_messages = self._to_langchain_messages(trimmed)
        excerpts = [hit.document.page_content for hit in hits]
        user = HumanMessage(content=build_user_prompt(question=question, excerpts=excerpts, mode=mode))
        return [system, *history_messages, user]

    def _trim_history(self, history: Sequence[ChatMessage]) -> list[ChatMessage]:
        """Keep history short so we don't blow the context window on long threads."""
        return list(history)[-self._settings.max_conversation_history :]

    def _to_langchain_messages(self, messages: Sequence[ChatMessage]) -> list[BaseMessage]:
        lc: list[BaseMessage] = []
        for m in messages:
            if m.role == MessageRole.USER:
                lc.append(HumanMessage(content=m.content))
            elif m.role == MessageRole.ASSISTANT:
                lc.append(AIMessage(content=m.content))
            elif m.role == MessageRole.SYSTEM:
                lc.append(SystemMessage(content=m.content))
        return lc

    def _rewrite_question(self, history: Sequence[ChatMessage], question: str) -> str:
        if not history:
            return question
        lc_messages: list[BaseMessage] = [SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT)]
        lc_messages.extend(self._to_langchain_messages(self._trim_history(history)))
        lc_messages.append(HumanMessage(content=f"New question: {question}"))
        try:
            response = self._utility.invoke(lc_messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite failed: %s", exc)
            return question
        rewritten = (response.content or question).strip().strip('"')
        return rewritten or question

    def _expand_query(self, question: str) -> list[str]:
        if self._settings.query_variations <= 1:
            return []
        structured = self._utility.with_structured_output(_QueryVariations)
        try:
            response: _QueryVariations = structured.invoke(
                [
                    SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Generate {self._settings.query_variations} variations of this query:\n{question}"
                        )
                    ),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query expansion failed: %s", exc)
            return []
        seen = {question.lower().strip()}
        out: list[str] = []
        for q in response.queries or []:
            normalized = q.strip()
            fingerprint = normalized.lower()
            if normalized and fingerprint not in seen:
                seen.add(fingerprint)
                out.append(normalized)
            if len(out) >= self._settings.query_variations:
                break
        return out

    def _build_citations(self, hits: Sequence[RetrievalHit]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int | None, str]] = set()
        for hit in hits:
            doc: Document = hit.document
            metadata = doc.metadata or {}
            source = str(metadata.get("source") or metadata.get("file_path") or "unknown")
            filename = Path(source).name if source and source != "unknown" else (
                metadata.get("filename") or "unknown"
            )
            raw_page = metadata.get("page", metadata.get("page_number"))
            if isinstance(raw_page, int):
                page: int | None = raw_page
            elif isinstance(raw_page, str) and raw_page.isdigit():
                page = int(raw_page)
            else:
                page = None
            if source.lower().endswith(".txt"):
                page = None
            content = doc.page_content.strip()
            if not content:
                continue
            identity = (source, page, content)
            if identity in seen:
                continue
            seen.add(identity)
            citations.append(
                Citation(
                    source=source,
                    filename=str(filename),
                    page=page,
                    content=content,
                    document_id=metadata.get("document_id"),
                    chunk_id=metadata.get("chunk_id"),
                    score=hit.score,
                    tag=metadata.get("tag"),
                )
            )
        return citations

    # ------- Chitchat / refusal paths (no retrieval) ----------
    def _handle_chitchat(
        self,
        conversation_id: UUID,
        history: Sequence[ChatMessage],
        question: str,
        mode: ChatMode,
        start: float,
    ) -> Iterable[dict[str, Any]]:
        yield {"status": "chitchat"}
        self._conversation_service.add_user_message(conversation_id, question)
        messages: list[BaseMessage] = [SystemMessage(content=CHITCHAT_SYSTEM_PROMPT)]
        messages.extend(self._to_langchain_messages(self._trim_history(history)))
        messages.append(HumanMessage(content=question))

        accumulated: list[str] = []
        final_usage: TokenUsage | None = None
        for delta, usage in self._utility.stream_with_usage(messages):
            if delta:
                accumulated.append(delta)
                yield {"delta": delta}
            if usage is not None:
                final_usage = usage

        answer = "".join(accumulated).strip()
        usage = final_usage or TokenUsage()
        latency_ms = int((time.perf_counter() - start) * 1000)
        if answer:
            self._conversation_service.append_assistant(
                conversation_id,
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=answer,
                    usage=usage,
                    latency_ms=latency_ms,
                    mode=mode.value,
                ),
            )
        yield {
            "usage": asdict(usage),
            "latency_ms": latency_ms,
            "cache_hit": False,
            "mode": mode.value,
            "intent": "chitchat",
        }
        self._usage.record(
            UsageEvent(
                conversation_id=str(conversation_id),
                mode="chitchat",
                model=self._utility.model_name,
                usage=usage,
                latency_ms=latency_ms,
                cache_hit=False,
            )
        )

    def _finalize_refusal(
        self,
        conversation_id: UUID,
        question: str,
        message: str,
        *,
        mode: ChatMode,
        start: float,
    ) -> Iterable[dict[str, Any]]:
        self._conversation_service.add_user_message(conversation_id, question)
        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = TokenUsage()
        self._conversation_service.append_assistant(
            conversation_id,
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=message,
                usage=usage,
                latency_ms=latency_ms,
                mode=mode.value,
            ),
        )
        yield {"status": "refused"}
        yield {"delta": message}
        yield {
            "usage": asdict(usage),
            "latency_ms": latency_ms,
            "cache_hit": False,
            "mode": mode.value,
            "intent": "refuse",
        }

    def _refusal_result(
        self,
        conversation_id: UUID,
        question: str,
        message: str,
        *,
        mode: ChatMode,
        start: float,
    ) -> ChatGenerationResult:
        self._conversation_service.add_user_message(conversation_id, question)
        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = TokenUsage()
        self._conversation_service.append_assistant(
            conversation_id,
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=message,
                usage=usage,
                latency_ms=latency_ms,
                mode=mode.value,
            ),
        )
        return ChatGenerationResult(
            answer=message,
            citations=[],
            usage=usage,
            latency_ms=latency_ms,
            mode=mode.value,
            refused=True,
        )

    def _chitchat_result(
        self,
        conversation_id: UUID,
        history: Sequence[ChatMessage],
        question: str,
        mode: ChatMode,
        start: float,
    ) -> ChatGenerationResult:
        self._conversation_service.add_user_message(conversation_id, question)
        messages: list[BaseMessage] = [SystemMessage(content=CHITCHAT_SYSTEM_PROMPT)]
        messages.extend(self._to_langchain_messages(self._trim_history(history)))
        messages.append(HumanMessage(content=question))
        response = self._utility.invoke(messages)
        answer = (response.content if isinstance(response, AIMessage) else str(response.content)).strip()
        usage = _usage_from_response(response) or TokenUsage()
        latency_ms = int((time.perf_counter() - start) * 1000)
        if answer:
            self._conversation_service.append_assistant(
                conversation_id,
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=answer,
                    usage=usage,
                    latency_ms=latency_ms,
                    mode=mode.value,
                ),
            )
        return ChatGenerationResult(
            answer=answer,
            citations=[],
            usage=usage,
            latency_ms=latency_ms,
            mode=mode.value,
        )

    def _log_retrieval_run(
        self, run_tree: Any | None, question: str, hits: Sequence[RetrievalHit]
    ) -> None:
        if not run_tree or not hasattr(run_tree, "create_child"):
            return
        try:
            child = run_tree.create_child(
                name="retrieval", run_type="retriever", inputs={"question": question}
            )
            child.end(outputs={"documents": [_hit_snapshot(h) for h in hits]})
        except Exception:  # pragma: no cover
            pass


def _hit_snapshot(hit: RetrievalHit) -> dict[str, Any]:
    meta = hit.document.metadata or {}
    return {
        "document_id": meta.get("document_id"),
        "filename": meta.get("filename"),
        "chunk_id": meta.get("chunk_id"),
        "score": round(hit.score, 4),
        "sources": list(hit.sources),
        "preview": hit.document.page_content[:200],
    }


def _usage_from_response(response: BaseMessage) -> TokenUsage | None:
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("personarag_usage")
    if isinstance(value, TokenUsage):
        return value
    return None
