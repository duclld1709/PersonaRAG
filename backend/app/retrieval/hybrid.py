"""Hybrid retriever: BM25 + vector similarity fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from langchain_core.documents import Document

from app.domain.interfaces.vector_store import VectorStore

from .bm25 import BM25Index


@dataclass
class RetrievalHit:
    """A single retrieval candidate after fusion."""

    document: Document
    score: float
    sources: tuple[str, ...]  # e.g. ("vector", "bm25")


class HybridRetriever:
    """Run BM25 and vector search in parallel and fuse the rankings."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        vector_search: str = "mmr",
    ) -> None:
        if vector_search not in ("mmr", "similarity"):
            raise ValueError("vector_search must be 'mmr' or 'similarity'")
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._rrf_k = rrf_k
        self._vector_search = vector_search

    def retrieve(self, queries: Iterable[str], *, k: int, fetch_k: int) -> list[RetrievalHit]:
        """Retrieve and fuse hits across several query variations.

        For each query we run both BM25 and vector search. RRF fuses ranks
        across every (query, retriever) list; the resulting score is boosted
        by `vector_weight` or `bm25_weight` depending on source, so operators
        can tune the hybrid mix without touching the fusion formula.
        """

        per_rank_score: defaultdict[str, float] = defaultdict(float)
        source_map: defaultdict[str, set[str]] = defaultdict(set)
        doc_lookup: dict[str, Document] = {}

        search_kwargs = {"k": k}
        if self._vector_search == "mmr":
            search_kwargs |= {"fetch_k": fetch_k, "lambda_mult": 0.5}
        retriever = (
            self._vector_store.as_retriever(search_type=self._vector_search, search_kwargs=search_kwargs)
            if self._vector_weight > 0
            else None
        )

        for query in queries:
            query = query.strip()
            if not query:
                continue

            # A zero weight disables a branch entirely (no embedding call), which is
            # how the eval measures BM25-only and dense-only retrieval.
            vector_hits: Sequence[Document] = (retriever.invoke(query) or []) if retriever else []
            for position, doc in enumerate(vector_hits, start=1):
                key = self._key(doc)
                doc_lookup[key] = doc
                per_rank_score[key] += self._vector_weight / (self._rrf_k + position)
                source_map[key].add("vector")

            bm25_hits = self._bm25.search(query, top_k=k) if self._bm25_weight > 0 else []
            for position, (doc, _score) in enumerate(bm25_hits, start=1):
                key = self._key(doc)
                doc_lookup[key] = doc
                per_rank_score[key] += self._bm25_weight / (self._rrf_k + position)
                source_map[key].add("bm25")

        ordered = sorted(per_rank_score.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievalHit(
                document=doc_lookup[key],
                score=score,
                sources=tuple(sorted(source_map[key])),
            )
            for key, score in ordered
        ]

    @staticmethod
    def _key(doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            return chunk_id
        return f"{doc.metadata.get('source', 'unknown')}:{hash(doc.page_content)}"
