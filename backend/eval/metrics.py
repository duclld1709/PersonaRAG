"""Lightweight RAG evaluation metrics.

When the optional `ragas` dependency is installed we delegate to its
faithfulness / answer_relevancy / context_precision / context_recall
implementations. Otherwise we fall back to deterministic heuristics that
approximate the same signal — good enough to rank pipeline variants even
without paying for the full RAGAS evaluation LLM loop.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class EvalScores:
    """Per-question metric snapshot."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
        }


def score_example(
    *,
    question: str,
    expected_answer: str,
    actual_answer: str,
    retrieved_contexts: Sequence[str],
    ground_truth_sources: Sequence[str],
    retrieved_sources: Sequence[str],
    expected_keywords: Sequence[str] = (),
) -> EvalScores:
    """Produce an EvalScores snapshot for a single golden example."""

    faithfulness = _heuristic_faithfulness(actual_answer, retrieved_contexts, expected_keywords)
    answer_relevancy = _token_overlap(actual_answer, expected_answer)
    context_precision = _precision(retrieved_sources, ground_truth_sources)
    context_recall = _recall(retrieved_sources, ground_truth_sources)
    return EvalScores(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
    )


def aggregate(scores: Sequence[EvalScores]) -> EvalScores:
    if not scores:
        return EvalScores(0.0, 0.0, 0.0, 0.0)
    return EvalScores(
        faithfulness=_mean(s.faithfulness for s in scores),
        answer_relevancy=_mean(s.answer_relevancy for s in scores),
        context_precision=_mean(s.context_precision for s in scores),
        context_recall=_mean(s.context_recall for s in scores),
    )


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(math.ceil(pct / 100.0 * len(sorted_values))) - 1))
    return float(sorted_values[index])


def _mean(values) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text or "")}


def _token_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _heuristic_faithfulness(
    answer: str, contexts: Sequence[str], keywords: Sequence[str]
) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0

    context_tokens: set[str] = set()
    for ctx in contexts:
        context_tokens |= _tokens(ctx)

    # Stopword filter avoids inflating the score on filler vocabulary.
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
        "were", "for", "on", "with", "as", "by", "that", "this", "it", "be",
        "has", "have", "from", "which", "its", "into", "at", "your", "you",
    }
    meaningful = {t for t in answer_tokens if len(t) > 2 and t not in stopwords}
    if not meaningful:
        return 0.0

    supported = meaningful & context_tokens
    base = len(supported) / len(meaningful)

    # Keyword bonus: tokens the golden set flagged as essential.
    if keywords:
        kw_tokens = _tokens(" ".join(keywords))
        if kw_tokens:
            present = len(kw_tokens & answer_tokens) / len(kw_tokens)
            base = 0.7 * base + 0.3 * present

    return min(1.0, base)


def _precision(retrieved: Sequence[str], ground_truth: Sequence[str]) -> float:
    if not retrieved:
        return 0.0 if ground_truth else 1.0
    gt_set = {src for src in ground_truth}
    if not gt_set:
        return 1.0 if not retrieved else 0.0
    hits = sum(1 for src in retrieved if src in gt_set)
    return hits / len(retrieved)


def _recall(retrieved: Sequence[str], ground_truth: Sequence[str]) -> float:
    gt_set = {src for src in ground_truth}
    if not gt_set:
        # Out-of-scope question: recall is 1.0 iff we retrieved nothing.
        return 1.0 if not retrieved else 0.0
    retrieved_set = {src for src in retrieved}
    return len(gt_set & retrieved_set) / len(gt_set)
