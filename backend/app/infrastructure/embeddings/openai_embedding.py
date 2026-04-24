"""OpenAI embeddings adapter with usage tracking."""

from __future__ import annotations

from threading import Lock
from typing import Iterable, Sequence

from langchain_openai import OpenAIEmbeddings

from app.domain.interfaces.embedding_client import EmbeddingClient
from app.domain.models.chat import TokenUsage
from app.infrastructure.llm.pricing import estimate_embedding_cost


class OpenAIEmbeddingClient(EmbeddingClient):
    """Wraps OpenAIEmbeddings to tally embedding token usage."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self._model_name = model_name
        self._client = OpenAIEmbeddings(model=model_name, api_key=api_key)
        self._cumulative = TokenUsage()
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Iterable[str]) -> Sequence[list[float]]:
        materialized = list(texts)
        vectors = self._client.embed_documents(materialized)
        approx_tokens = sum(_approx_tokens(text) for text in materialized)
        self._record(approx_tokens)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._client.embed_query(text)
        self._record(_approx_tokens(text))
        return vector

    def snapshot_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._cumulative.prompt_tokens,
                cost_usd=self._cumulative.cost_usd,
            )

    def _record(self, tokens: int) -> None:
        cost = estimate_embedding_cost(self._model_name, tokens=tokens)
        with self._lock:
            self._cumulative.prompt_tokens += tokens
            self._cumulative.cost_usd = round(self._cumulative.cost_usd + cost, 6)


def _approx_tokens(text: str) -> int:
    """Cheap token estimate using the common ~4-chars-per-token heuristic.

    The embeddings API does not return usage metadata through LangChain, so
    we estimate locally. For billing accuracy, swap in tiktoken; for cost
    dashboards this is within ~10% which is sufficient for observability.
    """

    if not text:
        return 0
    return max(1, len(text) // 4)
