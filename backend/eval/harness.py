"""Build an isolated PersonaRAG instance over the eval corpus.

Indexing goes through the production DocumentService and ingestion pipeline, and
answers through the production ChatService. Only the settings differ between
configurations.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from app.bootstrap import Providers, bootstrap_dependencies
from app.core.config import Settings
from app.domain.models.document import DocumentTag
from eval.dataset import corpus_files

WORK_DIR = Path(__file__).parent / "reports" / ".work"


@dataclass(frozen=True)
class PipelineConfig:
    """One ablation setting. ``vector_weight=0`` disables dense retrieval, ``bm25_weight=0`` BM25."""

    name: str
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    vector_search: str = "mmr"
    query_variations: int = 3
    rerank: bool = True
    description: str = ""

    @property
    def needs_embeddings(self) -> bool:
        return self.vector_weight > 0

    @property
    def needs_llm(self) -> bool:
        return self.rerank or self.query_variations > 1


CONFIGS: dict[str, PipelineConfig] = {
    c.name: c
    for c in (
        PipelineConfig("bm25", vector_weight=0, bm25_weight=1, query_variations=1, rerank=False,
                       description="BM25 only"),
        PipelineConfig("dense", vector_weight=1, bm25_weight=0, vector_search="similarity", query_variations=1,
                       rerank=False, description="embedding similarity only (v1 baseline)"),
        PipelineConfig("dense-mmr", vector_weight=1, bm25_weight=0, vector_search="mmr", query_variations=1,
                       rerank=False, description="embedding MMR only"),
        PipelineConfig("hybrid", vector_search="similarity", query_variations=1, rerank=False,
                       description="BM25 + similarity, RRF"),
        PipelineConfig("hybrid-mmr", query_variations=1, rerank=False, description="BM25 + MMR, RRF"),
        PipelineConfig("hybrid+mq", vector_search="similarity", query_variations=3, rerank=False,
                       description="hybrid + 3 LLM query variations"),
        PipelineConfig("hybrid+rerank", vector_search="similarity", query_variations=1, rerank=True,
                       description="hybrid + LLM reranker"),
        PipelineConfig("full", vector_search="mmr", query_variations=3, rerank=True,
                       description="production default: hybrid-MMR + multi-query + rerank"),
    )
}  # fmt: skip


def build_app(config: PipelineConfig, providers: Providers, *, k: int = 5, fetch_k: int = 20) -> FastAPI:
    """A fresh app (own SQLite + Chroma under eval/reports/.work) with the corpus indexed."""

    # A fresh directory per app: Chroma caches clients by path within a process, so a
    # reused (deleted and recreated) path ends up read-only.
    workdir = WORK_DIR / f"{config.name}-{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True)
    settings = Settings(
        DATABASE_URL=f"sqlite:///{workdir / 'eval.db'}",
        RETRIEVAL_K=k,
        RETRIEVAL_FETCH_K=fetch_k,
        RERANK_TOP_N=k,
        HYBRID_VECTOR_WEIGHT=config.vector_weight,
        HYBRID_BM25_WEIGHT=config.bm25_weight,
        VECTOR_SEARCH=config.vector_search,
        QUERY_VARIATIONS=config.query_variations,
        ENABLE_SEMANTIC_CACHE=False,  # every golden question is asked once; caching would only add noise
        ENABLE_MODERATION=False,
    )
    settings.docs_path = workdir / "docs"
    settings.vectorstore_path = workdir / "chroma"
    settings.docs_path.mkdir()

    app = FastAPI()
    bootstrap_dependencies(app, settings, providers=providers)
    if not config.rerank:
        app.state.chat_service._reranker = _TruncatingReranker()
    for path in corpus_files():
        app.state.document_service.upsert_document(
            filename=path.name, content=path.read_text(encoding="utf-8"), tag=DocumentTag.TEXTBOOK
        )
    return app


def clean_workdir() -> None:
    shutil.rmtree(WORK_DIR, ignore_errors=True)


class _TruncatingReranker:
    def rerank(self, question, hits, *, top_n):  # noqa: ARG002
        return list(hits[:top_n])
