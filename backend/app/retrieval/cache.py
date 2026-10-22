"""Semantic response cache keyed on query embeddings.

Students often ask the same question several ways ("what is overfitting",
"explain overfitting"). A cosine-similarity cache returns the earlier answer
without retrieval or generation.

Correctness rules (v1 had none of these):

* **Scope.** An entry only matches requests with the same scope (the response mode),
  so a cached *Quiz* never answers an *Explain* request.
* **First turn only.** The caller only consults the cache for standalone questions.
  "And the second one?" depends on the conversation, so it must never hit an answer
  produced in another conversation.
* **Invalidation.** :meth:`clear` is called whenever the indexed corpus changes, so
  answers citing deleted or edited documents are never served.
* **Live threshold.** The similarity threshold is read per call, so changing it in the
  settings panel takes effect immediately.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from time import time
from typing import Callable, List, Sequence

from app.domain.interfaces.embedding_client import EmbeddingClient
from app.domain.models.chat import Citation


@dataclass
class CachedAnswer:
    answer: str
    citations: List[Citation]
    created_at: float
    similarity: float = 1.0


@dataclass
class _CacheEntry:
    scope: str
    vector: list[float]
    norm: float
    value: CachedAnswer


class SemanticResponseCache:
    """Bounded (FIFO + TTL) semantic cache over question embeddings."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        threshold: float | Callable[[], float] = 0.95,
        max_entries: int = 256,
        ttl_seconds: int = 60 * 60 * 24,
    ) -> None:
        self._embeddings = embedding_client
        self._threshold = threshold if callable(threshold) else (lambda: threshold)
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._entries: list[_CacheEntry] = []
        self._lock = threading.Lock()

    def lookup(self, question: str, *, scope: str = "") -> CachedAnswer | None:
        if not question.strip():
            return None
        vector = self._embeddings.embed_query(question)
        norm = _norm(vector)
        if norm == 0:
            return None
        now = time()
        with self._lock:
            self._entries = [e for e in self._entries if e.value.created_at >= now - self._ttl_seconds]
            best_score, best = 0.0, None
            for entry in self._entries:
                if entry.scope != scope or entry.norm == 0:
                    continue
                score = _dot(entry.vector, vector) / (entry.norm * norm)
                if score > best_score:
                    best_score, best = score, entry
        if best is not None and best_score >= self._threshold():
            v = best.value
            return CachedAnswer(v.answer, list(v.citations), v.created_at, similarity=round(best_score, 4))
        return None

    def store(self, question: str, answer: str, citations: Sequence[Citation], *, scope: str = "") -> None:
        if not question.strip() or not answer.strip():
            return
        vector = self._embeddings.embed_query(question)
        norm = _norm(vector)
        if norm == 0:
            return
        entry = _CacheEntry(scope, vector, norm, CachedAnswer(answer, list(citations), time()))
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
