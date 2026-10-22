"""Chat orchestration for the research-assistant RAG workflow.

One pipeline, two entry points. :meth:`ChatService.stream_answer` yields SSE events,
and :meth:`ChatService.generate_answer` consumes the same generator. The eval harness
therefore measures exactly the code path users hit. v1 had two hand-maintained copies
that had already drifted apart.

    moderation → route ─┬─ refuse   → templated refusal
                        ├─ chitchat → utility model, no retrieval
                        └─ retrieve → [first turn: semantic cache] → rewrite → expand
                                      → hybrid retrieve → rerank → generate → cache
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.config import Settings
from app.core.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    MODERATION_MESSAGE,
    NOT_FOUND_PHRASE,
    QUERY_EXPANSION_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
    RESEARCH_ASSISTANT_SYSTEM_PROMPT,
    ChatMode,
    build_user_prompt,
)
from app.core.usage import UsageTracker, tracking
from app.domain.interfaces.llm_client import LLMClient
from app.domain.models.chat import ChatMessage, Citation, MessageRole, TokenUsage
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


class _QueryVariations(BaseModel):
    queries: list[str]


@dataclass
class ChatGenerationResult:
    """A finished answer plus everything needed to evaluate it."""

    answer: str
    citations: list[Citation]
    usage: TokenUsage
    latency_ms: int
    mode: str
    cache_hit: bool = False
    refused: bool = False
    intent: str = "retrieve"
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    contexts: list[str] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        conversation_service: ConversationService,
        chat_client: LLMClient,
        utility_client: LLMClient,
        moderation_client: Any,
        retriever: HybridRetriever,
        reranker: LLMReranker,
        router: QueryRouter,
        cache: SemanticResponseCache,
        usage_repository: SqliteUsageRepository,
    ) -> None:
        self._settings = settings
        self._conversation_service = conversation_service
        self._chat = chat_client
        self._utility = utility_client
        self._moderation = moderation_client
        self._retriever = retriever
        self._reranker = reranker
        self._router = router
        self._cache = cache
        self._usage = usage_repository

    def set_clients(self, *, chat_client: LLMClient, utility_client: LLMClient) -> None:
        """Swap models at runtime (settings panel)."""

        self._chat = chat_client
        self._utility = utility_client
        self._reranker._client = utility_client
        self._router._client = utility_client

    # ------------------------------------------------------------------ entry points

    def stream_answer(
        self, conversation_id: UUID, question: str, *, mode: ChatMode = ChatMode.ANSWER
    ) -> Iterator[dict[str, Any]]:
        """Yield ``status``/``delta``/``citations`` events, then one final ``usage`` event."""

        start = time.perf_counter()
        tracker = UsageTracker()
        conversation = self._conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")
        history = list(conversation.messages)
        run = _Run(conversation_id, question, mode, start, tracker)

        if self._settings.enable_moderation:
            with tracking(tracker, "moderation"):
                flagged = self._moderation.check(question).flagged
            if flagged:
                yield {"status": "refused"}
                yield {"delta": MODERATION_MESSAGE}
                yield from self._finish(run, MODERATION_MESSAGE, intent="refuse", refused=True)
                return

        yield {"status": "routing"}
        with tracking(tracker, "router"):
            intent = self._router.classify(question)

        if intent == QueryIntent.REFUSE:
            yield {"status": "refused"}
            yield {"delta": REFUSAL_MESSAGE}
            yield from self._finish(run, REFUSAL_MESSAGE, intent="refuse", refused=True)
            return

        if intent == QueryIntent.CHITCHAT:
            yield {"status": "chitchat"}
            messages = [
                SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
                *self._to_langchain_messages(self._trim_history(history)),
                HumanMessage(content=question),
            ]
            answer = yield from self._stream_llm(self._utility, messages, tracker, "chitchat")
            yield from self._finish(run, answer, intent="chitchat", model=self._utility.model_name)
            return

        # Only standalone (first-turn) questions may use the cache: a follow-up's
        # meaning depends on this conversation.
        use_cache = self._settings.enable_semantic_cache and not history
        if use_cache:
            with tracking(tracker, "cache"):
                cached = self._cache.lookup(question, scope=mode.value)
            if cached:
                yield {"status": "cache_hit"}
                yield {"delta": cached.answer}
                yield from self._finish(run, cached.answer, cached.citations, cache_hit=True)
                return

        yield {"status": "retrieving"}
        with tracking(tracker, "rewrite"):
            search_query = self._rewrite_question(history, question)
        with tracking(tracker, "expand"):
            expansions = self._expand_query(search_query)
        with tracking(tracker, "retrieve"):
            hits = self._retriever.retrieve(
                [search_query, *expansions], k=self._settings.retrieval_k, fetch_k=self._settings.retrieval_fetch_k
            )

        yield {"status": "reranking"}
        with tracking(tracker, "rerank"):
            top_hits = self._reranker.rerank(question, hits, top_n=self._settings.rerank_top_n)
        citations = self._build_citations(top_hits)

        yield {"status": "generating"}
        # Internal event (not forwarded over SSE): the exact excerpts the model sees,
        # numbered as in the prompt. The eval judge scores against these.
        yield {"contexts": [hit.document.page_content for hit in top_hits]}
        messages = self._build_prompt(history, question, top_hits, mode)
        answer = yield from self._stream_llm(self._chat, messages, tracker, "generate")

        if use_cache and answer and NOT_FOUND_PHRASE.lower() not in answer.lower():
            with tracking(tracker, "cache"):
                self._cache.store(question, answer, citations, scope=mode.value)
        yield from self._finish(run, answer, citations)

    def generate_answer(
        self, conversation_id: UUID, question: str, *, mode: ChatMode = ChatMode.ANSWER
    ) -> ChatGenerationResult:
        """Non-streaming wrapper over :meth:`stream_answer` (REST + eval harness)."""

        parts: list[str] = []
        citations: list[Citation] = []
        contexts: list[str] = []
        final: dict[str, Any] = {}
        for event in self.stream_answer(conversation_id, question, mode=mode):
            if "delta" in event:
                parts.append(event["delta"])
            if "contexts" in event:
                contexts = event["contexts"]
            if "citations" in event:
                citations = [Citation(**c) for c in event["citations"]]
            if "usage" in event:
                final = event
        return ChatGenerationResult(
            answer="".join(parts).strip(),
            citations=citations,
            usage=TokenUsage(**final["usage"]),
            latency_ms=final["latency_ms"],
            mode=final["mode"],
            cache_hit=final["cache_hit"],
            refused=final["intent"] == "refuse",
            intent=final["intent"],
            cost_breakdown=final["cost_breakdown"],
            contexts=contexts,
        )

    # ------------------------------------------------------------------ pipeline steps

    def _stream_llm(
        self, client: LLMClient, messages: list[BaseMessage], tracker: UsageTracker, stage: str
    ) -> Iterator[dict[str, Any]]:
        """Stream deltas; returns the full text. Usage is added explicitly because the
        tracker context cannot span ``yield`` (see app.core.usage)."""

        parts: list[str] = []
        for delta, usage in client.stream_with_usage(messages):
            if delta:
                parts.append(delta)
                yield {"delta": delta}
            if usage is not None:
                tracker.add(client.model_name, usage, stage)
        return "".join(parts).strip()

    def _finish(
        self,
        run: _Run,
        answer: str,
        citations: Sequence[Citation] = (),
        *,
        intent: str = "retrieve",
        cache_hit: bool = False,
        refused: bool = False,
        model: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Persist the exchange, record usage per model, emit the final events."""

        latency_ms = int((time.perf_counter() - run.start) * 1000)
        total = run.tracker.total
        self._conversation_service.add_user_message(run.conversation_id, run.question)
        if answer:
            self._conversation_service.append_assistant(
                run.conversation_id,
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=answer,
                    citations=list(citations) or None,
                    usage=total,
                    latency_ms=latency_ms,
                    mode=run.mode.value,
                ),
            )
        self._record_usage(run, latency_ms, cache_hit, intent, model or self._chat.model_name)

        if citations:
            yield {"citations": [asdict(c) for c in citations]}
        yield {
            "usage": asdict(total),
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "mode": run.mode.value,
            "intent": intent,
            "refused": refused,
            "cost_breakdown": run.tracker.breakdown(),
        }

    def _record_usage(self, run: _Run, latency_ms: int, cache_hit: bool, intent: str, primary_model: str) -> None:
        """One row per model so the per-model cost summary is accurate. The row for the
        primary model carries latency and cache flag; request_id groups the rows."""

        mode = run.mode.value if intent == "retrieve" else intent
        by_model = dict(run.tracker.by_model) or {primary_model: TokenUsage()}
        primary = primary_model if primary_model in by_model else next(iter(by_model))
        for model, usage in by_model.items():
            is_primary = model == primary
            self._usage.record(
                UsageEvent(
                    conversation_id=str(run.conversation_id),
                    mode=mode,
                    model=model,
                    usage=usage,
                    latency_ms=latency_ms if is_primary else None,
                    cache_hit=cache_hit and is_primary,
                    request_id=run.request_id,
                )
            )

    def _build_prompt(
        self, history: Sequence[ChatMessage], question: str, hits: Sequence[RetrievalHit], mode: ChatMode
    ) -> list[BaseMessage]:
        excerpts = [hit.document.page_content for hit in hits]
        return [
            SystemMessage(content=RESEARCH_ASSISTANT_SYSTEM_PROMPT),
            *self._to_langchain_messages(self._trim_history(history)),
            HumanMessage(content=build_user_prompt(question=question, excerpts=excerpts, mode=mode)),
        ]

    def _trim_history(self, history: Sequence[ChatMessage]) -> list[ChatMessage]:
        return list(history)[-self._settings.max_conversation_history :]

    @staticmethod
    def _to_langchain_messages(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
        kinds = {MessageRole.USER: HumanMessage, MessageRole.ASSISTANT: AIMessage, MessageRole.SYSTEM: SystemMessage}
        return [kinds[m.role](content=m.content) for m in messages if m.role in kinds]

    def _rewrite_question(self, history: Sequence[ChatMessage], question: str) -> str:
        if not history:
            return question
        messages: list[BaseMessage] = [
            SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
            *self._to_langchain_messages(self._trim_history(history)),
            HumanMessage(content=f"New question: {question}"),
        ]
        try:
            response = self._utility.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite failed: %s", exc)
            return question
        rewritten = str(response.content or "").strip().strip('"')
        return rewritten or question

    def _expand_query(self, question: str) -> list[str]:
        n = self._settings.query_variations
        if n <= 1:
            return []
        try:
            response = self._utility.invoke_structured(
                [
                    SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
                    HumanMessage(content=f"Generate {n} variations of this query:\n{question}"),
                ],
                _QueryVariations,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query expansion failed: %s", exc)
            return []
        seen = {question.lower().strip()}
        out: list[str] = []
        for q in response.queries or []:
            normalized = q.strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                out.append(normalized)
        return out[:n]

    @staticmethod
    def _build_citations(hits: Sequence[RetrievalHit]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int | None, str]] = set()
        for hit in hits:
            doc: Document = hit.document
            metadata = doc.metadata or {}
            source = str(metadata.get("source") or metadata.get("file_path") or "unknown")
            filename = metadata.get("filename") or (Path(source).name if source != "unknown" else "unknown")
            raw_page = metadata.get("page", metadata.get("page_number"))
            page = raw_page if isinstance(raw_page, int) else int(raw_page) if str(raw_page).isdigit() else None
            content = doc.page_content.strip()
            if not content or (source, page, content) in seen:
                continue
            seen.add((source, page, content))
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


@dataclass
class _Run:
    conversation_id: UUID
    question: str
    mode: ChatMode
    start: float
    tracker: UsageTracker
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
