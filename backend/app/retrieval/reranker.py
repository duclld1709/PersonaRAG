"""LLM-based reranker using a small OpenAI model with structured output.

A dedicated rerank step is what takes recall-heavy retrieval (many candidates)
and turns it into precision-heavy context (the handful the main LLM sees).
Using an OpenAI model keeps the stack single-vendor; using the cheap model
keeps the rerank cost at <10% of the main generation cost.
"""

from __future__ import annotations

from logging import getLogger
from typing import Sequence

from pydantic import BaseModel, Field

from app.core.prompts import RERANK_SYSTEM_PROMPT
from app.infrastructure.llm.openai_chat import OpenAIUtilityClient
from langchain_core.messages import HumanMessage, SystemMessage

from .hybrid import RetrievalHit

logger = getLogger(__name__)


class _RerankItem(BaseModel):
    index: int = Field(ge=1)
    score: float = Field(ge=0.0, le=10.0)


class _RerankResponse(BaseModel):
    rankings: list[_RerankItem]


class LLMReranker:
    """Score candidate excerpts for a question using a small chat model."""

    def __init__(self, client: OpenAIUtilityClient) -> None:
        self._client = client

    def rerank(self, question: str, hits: Sequence[RetrievalHit], *, top_n: int) -> list[RetrievalHit]:
        if not hits:
            return []
        # Short-circuit if we already have <= top_n candidates; rerank would
        # only ever shrink the list, and we want to avoid the extra API call.
        if len(hits) <= top_n:
            return list(hits)

        excerpts_block = "\n\n".join(
            f"[{i}] {hit.document.page_content.strip()[:800]}"
            for i, hit in enumerate(hits, start=1)
        )
        user_message = (
            f"Question: {question.strip()}\n\n"
            f"Candidates:\n{excerpts_block}\n\n"
            "Score each candidate 0-10 on usefulness for answering."
        )

        structured = self._client.with_structured_output(_RerankResponse)
        try:
            response: _RerankResponse = structured.invoke(
                [SystemMessage(content=RERANK_SYSTEM_PROMPT), HumanMessage(content=user_message)]
            )
        except Exception as exc:  # noqa: BLE001 — fall back to original ordering
            logger.warning("Reranker call failed, using pre-rerank ordering: %s", exc)
            return list(hits[:top_n])

        scored: dict[int, float] = {}
        for item in response.rankings:
            if 1 <= item.index <= len(hits):
                scored[item.index - 1] = float(item.score)

        # Blend rerank score with the original fusion score so near-ties from
        # the LLM still respect upstream hybrid evidence.
        blended: list[tuple[int, float]] = []
        for idx, hit in enumerate(hits):
            llm_score = scored.get(idx, 0.0) / 10.0
            blended.append((idx, 0.7 * llm_score + 0.3 * hit.score))
        blended.sort(key=lambda item: item[1], reverse=True)
        return [hits[idx] for idx, _ in blended[:top_n]]
