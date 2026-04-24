"""Run PersonaRAG against the golden Q&A set across pipeline configurations.

Produces:
  - eval/reports/<config>.json  per-config results
  - eval/reports/summary.md     README-ready Markdown table

Configs:
  baseline         vector-only retrieval, no rerank, no cache
  hybrid           BM25 + vector + RRF, no rerank, no cache
  hybrid+rerank    hybrid + LLM reranker
  full             hybrid + rerank + semantic cache  (production default)

Usage:
    python -m eval.run_eval                    # all configs, live
    python -m eval.run_eval --configs full     # one config
    python -m eval.run_eval --dry-run          # deterministic placeholder run
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from app.core.config import get_settings
from app.core.prompts import ChatMode
from app.domain.models.document import DocumentTag
from app.infrastructure.persistence.database import create_database
from app.infrastructure.persistence.sqlite_conversation_repository import (
    SqliteConversationRepository,
)
from app.infrastructure.persistence.sqlite_document_repository import SqliteDocumentRepository
from app.infrastructure.persistence.sqlite_usage_repository import SqliteUsageRepository
from app.infrastructure.storage import LocalFileStorage
from app.infrastructure.vectorstores import ChromaVectorStore
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.retrieval import BM25Index, HybridRetriever, LLMReranker, QueryRouter, SemanticResponseCache
from app.services import ChatService, ConversationService, DocumentService

from .metrics import EvalScores, aggregate, percentile, score_example

EVAL_DIR = Path(__file__).parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_FILE = EVAL_DIR / "golden_qa.jsonl"
REPORT_DIR = EVAL_DIR / "reports"

CONFIGS = {
    "baseline": {"hybrid": False, "rerank": False, "cache": False},
    "hybrid": {"hybrid": True, "rerank": False, "cache": False},
    "hybrid+rerank": {"hybrid": True, "rerank": True, "cache": False},
    "full": {"hybrid": True, "rerank": True, "cache": True},
}


@dataclass
class QuestionResult:
    id: str
    question: str
    actual_answer: str
    retrieved_sources: list[str]
    latency_ms: int
    tokens: int
    cost_usd: float
    cache_hit: bool
    scores: EvalScores

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "scores": self.scores.as_dict(),
        }


# ---------------------------------------------------------------------------
# Dry-run support
# ---------------------------------------------------------------------------

def _dry_run_result(example: dict, config_name: str) -> QuestionResult:
    """Deterministic placeholder so CI can verify the harness without API calls."""

    seed = sum(ord(c) for c in example["id"]) + sum(ord(c) for c in config_name)
    base = {
        "baseline": 0.72,
        "hybrid": 0.80,
        "hybrid+rerank": 0.85,
        "full": 0.88,
    }.get(config_name, 0.75)
    wobble = ((seed % 7) - 3) * 0.01
    faithfulness = max(0.0, min(1.0, base + wobble))
    scores = EvalScores(
        faithfulness=faithfulness,
        answer_relevancy=faithfulness - 0.05,
        context_precision=faithfulness - 0.02,
        context_recall=faithfulness - 0.08,
    )
    return QuestionResult(
        id=example["id"],
        question=example["question"],
        actual_answer="(dry-run placeholder answer)",
        retrieved_sources=list(example.get("ground_truth_sources", [])),
        latency_ms=1800 - {"baseline": 0, "hybrid": 200, "hybrid+rerank": 400, "full": -100}.get(config_name, 0),
        tokens=1200,
        cost_usd=0.0042,
        cache_hit=False,
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------

def _build_chat_service(config_name: str) -> ChatService:
    """Construct a ChatService wired for the requested config.

    Each config uses an isolated SQLite file and Chroma collection so runs
    don't pollute each other.
    """

    from app.infrastructure.embeddings import OpenAIEmbeddingClient
    from app.infrastructure.llm import OpenAIChatClient, OpenAIUtilityClient
    from app.infrastructure.moderation import OpenAIModerationClient

    settings = get_settings()
    run_root = REPORT_DIR / "workdir" / config_name
    run_root.mkdir(parents=True, exist_ok=True)
    db_path = run_root / "eval.db"
    chroma_path = run_root / "chroma_db"
    docs_path = run_root / "docs"

    if db_path.exists():
        db_path.unlink()
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    if docs_path.exists():
        shutil.rmtree(docs_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    db = create_database(f"sqlite:///{db_path}", base_dir=Path.cwd())
    convo_repo = SqliteConversationRepository(db)
    doc_repo = SqliteDocumentRepository(db)
    usage_repo = SqliteUsageRepository(db)

    embedder = OpenAIEmbeddingClient(settings.embedding_model, api_key=settings.openai_api_key)
    chat_client = OpenAIChatClient(settings.chat_model, api_key=settings.openai_api_key)
    utility_client = OpenAIUtilityClient(settings.utility_model, api_key=settings.openai_api_key)
    moderation = OpenAIModerationClient(settings.openai_api_key, model=settings.moderation_model)

    vector_store = ChromaVectorStore(chroma_path, embedder)
    bm25 = BM25Index()
    pipeline = DocumentIngestionPipeline(settings=settings, vector_store=vector_store)
    pipeline.set_on_change(lambda: bm25.rebuild(list(vector_store.get())))
    storage = LocalFileStorage(docs_path)
    document_service = DocumentService(doc_repo, storage, pipeline)

    for source in CORPUS_DIR.glob("*.md"):
        document_service.upsert_document(
            filename=source.name, content=source.read_text(encoding="utf-8"), tag=DocumentTag.TEXTBOOK
        )
    bm25.rebuild(list(vector_store.get()))

    cfg = CONFIGS[config_name]
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25,
        vector_weight=settings.hybrid_vector_weight if cfg["hybrid"] else 1.0,
        bm25_weight=settings.hybrid_bm25_weight if cfg["hybrid"] else 0.0,
        rrf_k=settings.rrf_k,
    )
    reranker = LLMReranker(utility_client) if cfg["rerank"] else _NoopReranker()
    cache = (
        SemanticResponseCache(embedder, threshold=settings.semantic_cache_threshold)
        if cfg["cache"]
        else _NoopCache()
    )
    router = _AllRetrieveRouter()
    convo_service = ConversationService(convo_repo, settings)

    return ChatService(
        settings=settings,
        conversation_service=convo_service,
        vector_store=vector_store,
        chat_client=chat_client,
        utility_client=utility_client,
        embedding_client=embedder,
        moderation_client=moderation,
        retriever=retriever,
        reranker=reranker,  # type: ignore[arg-type]
        router=router,  # type: ignore[arg-type]
        cache=cache,  # type: ignore[arg-type]
        usage_repository=usage_repo,
    )


class _NoopReranker:
    def rerank(self, question, hits, *, top_n):  # noqa: D401, ARG002
        return list(hits[:top_n])


class _NoopCache:
    def lookup(self, question):  # noqa: D401, ARG002
        return None

    def store(self, question, answer, citations):  # noqa: D401, ARG002
        return None


class _AllRetrieveRouter:
    """Forces all golden questions down the retrieval path for fair comparison."""

    def classify(self, message: str):
        from app.retrieval.router import QueryIntent

        return QueryIntent.RETRIEVE


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_config(config_name: str, examples: Sequence[dict], *, dry_run: bool) -> list[QuestionResult]:
    if dry_run:
        return [_dry_run_result(ex, config_name) for ex in examples]

    service = _build_chat_service(config_name)
    convo_service: ConversationService = service._conversation_service  # type: ignore[attr-defined]
    results: list[QuestionResult] = []

    for example in examples:
        conversation = convo_service.create_conversation(title=f"eval-{example['id']}")
        start = time.perf_counter()
        outcome = service.generate_answer(
            conversation.id, example["question"], mode=ChatMode.ANSWER
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        retrieved_sources = sorted(
            {Path(c.source).name for c in outcome.citations if c.source}
        )
        scores = score_example(
            question=example["question"],
            expected_answer=example["expected_answer"],
            actual_answer=outcome.answer,
            retrieved_contexts=[c.content for c in outcome.citations],
            ground_truth_sources=example.get("ground_truth_sources", []),
            retrieved_sources=retrieved_sources,
            expected_keywords=example.get("keywords", []),
        )
        results.append(
            QuestionResult(
                id=example["id"],
                question=example["question"],
                actual_answer=outcome.answer,
                retrieved_sources=retrieved_sources,
                latency_ms=latency_ms,
                tokens=outcome.usage.total_tokens,
                cost_usd=outcome.usage.cost_usd,
                cache_hit=outcome.cache_hit,
                scores=scores,
            )
        )
    return results


def load_examples() -> list[dict]:
    examples: list[dict] = []
    with GOLDEN_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def summarize(config: str, results: Sequence[QuestionResult]) -> dict[str, Any]:
    scores = [r.scores for r in results]
    latencies = [float(r.latency_ms) for r in results]
    tokens = [r.tokens for r in results]
    costs = [r.cost_usd for r in results]
    agg = aggregate(scores)
    return {
        "config": config,
        "n": len(results),
        "scores": agg.as_dict(),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "tokens_avg": sum(tokens) / len(tokens) if tokens else 0,
        "cost_usd_avg": round(sum(costs) / len(costs), 6) if costs else 0.0,
    }


def write_report(summaries: Sequence[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PersonaRAG Evaluation Report",
        "",
        "| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall | p50 Latency | p95 Latency | Avg Tokens | Avg Cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in summaries:
        s = summary["scores"]
        lines.append(
            "| {config} | {f:.3f} | {ar:.3f} | {cp:.3f} | {cr:.3f} | {p50:.0f} ms | {p95:.0f} ms | {tk:.0f} | ${c:.4f} |".format(
                config=summary["config"],
                f=s["faithfulness"],
                ar=s["answer_relevancy"],
                cp=s["context_precision"],
                cr=s["context_recall"],
                p50=summary["latency_ms_p50"],
                p95=summary["latency_ms_p95"],
                tk=summary["tokens_avg"],
                c=summary["cost_usd_avg"],
            )
        )
    lines.append("")
    lines.append(f"_n = {summaries[0]['n']} golden questions._")
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PersonaRAG evaluation.")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--dry-run", action="store_true", help="Skip OpenAI calls; emit placeholder numbers.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_examples()
    summaries: list[dict[str, Any]] = []

    for config_name in args.configs:
        print(f"[eval] running config={config_name} n={len(examples)} dry_run={args.dry_run}")
        results = run_config(config_name, examples, dry_run=args.dry_run)
        summary = summarize(config_name, results)
        (REPORT_DIR / f"{config_name}.json").write_text(
            json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summaries.append(summary)
        print(f"[eval] done config={config_name}: {summary['scores']}")

    write_report(summaries)
    print(f"[eval] report written to {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    _ = uuid.uuid4()  # reserved for future per-run identifiers
    main()
