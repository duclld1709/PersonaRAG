"""BM25 sparse index rebuilt on demand from the vector store's chunks.

Kept lightweight: the rank_bm25 library is pure-Python but fast enough for
the student-materials scale (hundreds to low thousands of chunks). For
bigger corpora, swap for a proper inverted index (e.g. Elastic / Tantivy).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


@dataclass
class _Entry:
    document: Document
    tokens: list[str]


class BM25Index:
    """Thread-safe BM25 index over LangChain Documents."""

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    def rebuild(self, documents: Sequence[Document]) -> None:
        entries = [_Entry(doc, _tokenize(doc.page_content)) for doc in documents]
        bm25 = BM25Okapi([e.tokens for e in entries]) if entries else None
        with self._lock:
            self._entries = entries
            self._bm25 = bm25

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        with self._lock:
            if not self._bm25 or not self._entries:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(
                zip(self._entries, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            return [(entry.document, float(score)) for entry, score in ranked[:top_k] if score > 0]
