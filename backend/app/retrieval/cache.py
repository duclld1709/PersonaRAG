"""Semantic response cache keyed on query embeddings.

For a research assistant, students often ask the same question multiple ways
("what is overfitting", "explain overfitting", "overfitting definition"). A
cosine-similarity cache keyed on the query embedding gives near-instant,
$0 replies for these duplicates without the brittleness of exact-match keys.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from time import time
from typing import List, Sequence

from app.domain.interfaces.embedding_client import EmbeddingClient
from app.domain.models.chat import Citation


@dataclass
class CachedAnswer:
    """Materialised cache entry."""

    answer: str
    citations: List[Citation]
    created_at: float


@dataclass
class _CacheEntry:
    vector: list[float]
    norm: float
    value: CachedAnswer


class SemanticResponseCache:
    """FIFO-bounded semantic cache over question embeddings."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        threshold: float = 0.95,
        max_entries: int = 256,
        ttl_seconds: int = 60 * 60 * 24,
    ) -> None:
        self._embeddings = embedding_client
        self._threshold = threshold
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: list[_CacheEntry] = []
        self._lock = threading.Lock()

    def lookup(self, question: str) -> CachedAnswer | None:
        if not question.strip():
            return None
        vector = self._embeddings.embed_query(question)
        norm = _norm(vector)
        if norm == 0:
            return None
        now = time()
        with self._lock:
            self._evict_expired(now)
            best_score = 0.0
            best_entry: _CacheEntry | None = None
            for entry in self._entries:
                denom = entry.norm * norm
                if denom == 0:
                    continue
                score = _dot(entry.vector, vector) / denom
                if score > best_score:
                    best_score = score
                    best_entry = entry
            if best_entry and best_score >= self._threshold:
                return best_entry.value
        return None

    def store(self, question: str, answer: str, citations: Sequence[Citation]) -> None:
        if not question.strip() or not answer.strip():
            return
        vector = self._embeddings.embed_query(question)
        norm = _norm(vector)
        if norm == 0:
            return
        entry = _CacheEntry(
            vector=vector,
            norm=norm,
            value=CachedAnswer(answer=answer, citations=list(citations), created_at=time()),
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._ttl_seconds
        self._entries = [entry for entry in self._entries if entry.value.created_at >= cutoff]


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
