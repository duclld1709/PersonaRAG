"""Aggregation (with bootstrap confidence intervals) and report writing."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

RUNS_DIR = Path(__file__).parent / "reports" / "runs"


@dataclass
class Stat:
    mean: float
    lo: float
    hi: float
    n: int

    def fmt(self, pct: bool = False) -> str:
        if self.n == 0:
            return "–"
        f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}")
        return f"{f(self.mean)} [{f(self.lo)}, {f(self.hi)}]"


def bootstrap(values: Iterable[float | None], n_boot: int = 2000, seed: int = 0) -> Stat:
    """Mean with a 95% percentile-bootstrap interval (None values are skipped)."""

    xs = [float(v) for v in values if v is not None]
    if not xs:
        return Stat(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return Stat(mean(xs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1], len(xs))


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))]


def summarise_retrieval(records: list) -> dict:
    return {
        "n": len(records),
        **{m: bootstrap(getattr(r, m) for r in records).__dict__ for m in ("hit", "recall", "mrr", "ndcg")},
        "latency_p50_ms": percentile([r.latency_ms for r in records], 50),
        "cost_usd_mean": mean(r.cost_usd for r in records) if records else 0.0,
    }


def summarise_generation(records: list) -> dict:
    in_scope = [r for r in records if r.type in ("single", "multi")]
    oos = [r for r in records if r.type == "out_of_scope"]
    refuse = [r for r in records if r.type == "refuse"]
    stages: dict[str, list[float]] = {}
    for r in records:
        for stage, cost in r.cost_breakdown.items():
            stages.setdefault(stage, []).append(cost)
    return {
        "n": len(records),
        **{
            m: bootstrap(getattr(r, m) for r in in_scope).__dict__
            for m in ("faithfulness", "citation_precision", "correctness", "context_recall")
        },
        "false_abstention_rate": bootstrap(float(r.abstained) for r in in_scope).__dict__,
        "out_of_scope_correct": bootstrap(float(bool(r.behaved_correctly)) for r in oos).__dict__,
        "refusal_correct": bootstrap(float(bool(r.behaved_correctly)) for r in refuse).__dict__,
        "latency_p50_ms": percentile([r.latency_ms for r in records], 50),
        "latency_p95_ms": percentile([r.latency_ms for r in records], 95),
        "cost_usd_mean": mean(r.cost_usd for r in records) if records else 0.0,
        "cost_usd_by_stage_mean": {k: sum(v) / len(records) for k, v in sorted(stages.items())},
        "judge_cost_usd_total": sum(r.judge_cost_usd for r in records),
    }


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _s(d: dict, pct: bool = False) -> str:
    return Stat(**d).fmt(pct)


def markdown(meta: dict, retrieval: dict[str, dict], generation: dict[str, dict]) -> str:
    lines = [f"# PersonaRAG evaluation: {meta['run_id']}", ""]
    if meta["offline"]:
        lines += [
            "> **Smoke test.** Run with `--offline` (hash embeddings, heuristic LLM and judge). Only configurations "
            "marked *valid offline* (BM25) are real measurements; every other number only checks the harness.",
            "",
        ]
    lines += [
        f"- commit `{meta['git_sha']}`, corpus `{meta['corpus_fingerprint']}` ({meta['corpus_files']} files, "
        f"{meta['chunks']} chunks), golden set `{meta['golden_fingerprint']}` ({meta['n_questions']} questions)",
        f"- models: chat `{meta['models']['chat']}`, utility `{meta['models']['utility']}`, "
        f"embeddings `{meta['models']['embedding']}`, judge `{meta['models'].get('judge', '-')}`",
        f"- k = {meta['k']}; values are means with 95% bootstrap CIs",
        "",
    ]
    if retrieval:
        lines += [
            "## Retrieval (in-scope questions)",
            "",
            "| config | Hit@k | Recall@k | MRR | nDCG@k | p50 latency | cost / query |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, s in retrieval.items():
            flag = " *(valid offline)*" if s.get("valid_offline") and meta["offline"] else ""
            lines.append(
                f"| {name}{flag} | {_s(s['hit'], True)} | {_s(s['recall'], True)} | {_s(s['mrr'])} | {_s(s['ndcg'])} | "
                f"{s['latency_p50_ms']:.0f} ms | ${s['cost_usd_mean']:.5f} |"
            )
        lines.append("")
    if generation:
        lines += [
            "## Answers (end to end)",
            "",
            "| config | faithfulness | citation precision | correctness | context recall | false abstain | OOS ok | "
            "refusal ok | p50 / p95 latency | cost / question |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, s in generation.items():
            lines.append(
                f"| {name} | {_s(s['faithfulness'], True)} | {_s(s['citation_precision'], True)} | "
                f"{_s(s['correctness'], True)} | {_s(s['context_recall'], True)} | "
                f"{_s(s['false_abstention_rate'], True)} | {_s(s['out_of_scope_correct'], True)} | "
                f"{_s(s['refusal_correct'], True)} | {s['latency_p50_ms']:.0f} / {s['latency_p95_ms']:.0f} ms | "
                f"${s['cost_usd_mean']:.4f} |"
            )
        lines += ["", "Mean cost per question by pipeline stage (USD):", ""]
        for name, s in generation.items():
            parts = ", ".join(f"{k} {v:.5f}" for k, v in s["cost_usd_by_stage_mean"].items())
            lines.append(f"- **{name}**: {parts} (judge total ${s['judge_cost_usd_total']:.3f})")
    return "\n".join(lines) + "\n"


def write_run(meta: dict, retrieval_records: dict, generation_records: dict, retrieval: dict, generation: dict) -> Path:
    out = RUNS_DIR / meta["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for kind, groups in (("retrieval", retrieval_records), ("generation", generation_records)):
        for name, records in groups.items():
            path = out / f"{kind}-{name}.jsonl"
            path.write_text(
                "".join(json.dumps(r.as_dict(), ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
            )
    (out / "summary.json").write_text(
        json.dumps({"retrieval": retrieval, "generation": generation}, indent=2), encoding="utf-8"
    )
    (out / "summary.md").write_text(markdown(meta, retrieval, generation), encoding="utf-8")
    return out
