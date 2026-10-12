"""Retrieval metrics against evidence-labelled golden questions.

For each in-scope question we take the top-k chunks the pipeline would put in the
prompt (after optional query expansion and reranking) and compute

* **Hit@k**: at least one retrieved chunk contains evidence
* **Recall@k**: share of the question's evidence quotes covered by the top-k chunks
  (matters for multi-part questions)
* **MRR**: 1 / rank of the first relevant chunk
* **nDCG@k**: binary relevance, ideal ranking from all relevant chunks in the index
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from app.core.usage import UsageTracker, tracking
from eval.dataset import GoldenQuestion, covers


@dataclass
class RetrievalRecord:
    id: str
    type: str
    hit: float
    recall: float
    mrr: float
    ndcg: float
    latency_ms: float
    cost_usd: float
    ranks: list[int]  # 1-based ranks of relevant chunks in the top-k

    def as_dict(self) -> dict:
        return asdict(self)


def score_ranking(chunks: list[str], evidence: tuple[str, ...], n_relevant_in_index: int, k: int) -> dict[str, float]:
    top = chunks[:k]
    relevant = [any(covers(c, e) for e in evidence) for c in top]
    covered = {e for e in evidence if any(covers(c, e) for c in top)}
    first = next((i for i, r in enumerate(relevant, 1) if r), None)
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(relevant, 1) if r)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(n_relevant_in_index, k) + 1))
    return {
        "hit": float(any(relevant)),
        "recall": len(covered) / len(evidence) if evidence else 0.0,
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "ranks": [i for i, r in enumerate(relevant, 1) if r],
    }


def evaluate_retrieval(app, questions: list[GoldenQuestion], *, k: int) -> list[RetrievalRecord]:
    """Run the app's own retrieval stages (expansion → hybrid → rerank) for each question."""

    chat = app.state.chat_service
    all_chunks = [d.page_content for d in app.state.vector_store.get()]
    records = []
    for q in questions:
        if not q.in_scope:
            continue
        tracker = UsageTracker()
        start = time.perf_counter()
        with tracking(tracker, "retrieval"):
            expansions = chat._expand_query(q.question)
            hits = chat._retriever.retrieve(
                [q.question, *expansions], k=k, fetch_k=app.state.settings.retrieval_fetch_k
            )
            hits = chat._reranker.rerank(q.question, hits, top_n=k)
        latency = (time.perf_counter() - start) * 1000
        n_relevant = sum(1 for c in all_chunks if any(covers(c, e) for e in q.evidence))
        scores = score_ranking([h.document.page_content for h in hits], q.evidence, n_relevant, k)
        records.append(
            RetrievalRecord(q.id, q.type, latency_ms=round(latency, 1), cost_usd=tracker.total.cost_usd, **scores)
        )
    return records
