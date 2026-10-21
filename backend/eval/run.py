"""Run the PersonaRAG evaluation.

    python -m eval.fetch_corpus                       # once: download the textbook corpus
    python -m eval.run retrieval                      # retrieval ablation (embeddings + utility LLM)
    python -m eval.run generation --configs dense full --limit 40
    python -m eval.run all

    # no API key: BM25 numbers are real, everything else is a harness smoke test
    python -m eval.run all --offline
    python -m eval.run retrieval --offline --configs bm25 --min-hit-rate 0.8   # CI quality gate

Each run writes ``eval/reports/runs/<run id>/`` with per-question JSONL, summary.json and
summary.md, plus the commit, corpus/golden-set fingerprints and model names.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time

from app.bootstrap import openai_providers
from app.core.config import get_settings
from app.infrastructure.llm import OpenAIChatClient
from eval.dataset import GOLDEN_FILE, corpus_files, fingerprint, load_golden
from eval.generation import evaluate_generation
from eval.harness import CONFIGS, build_app, clean_workdir
from eval.report import git_sha, summarise_generation, summarise_retrieval, write_run
from eval.retrieval import evaluate_retrieval

RETRIEVAL_DEFAULT = ["bm25", "dense", "dense-mmr", "hybrid", "hybrid-mmr", "hybrid+mq", "hybrid+rerank", "full"]
GENERATION_DEFAULT = ["dense", "hybrid", "hybrid+rerank", "full"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["retrieval", "generation", "all"])
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS))
    p.add_argument("--offline", action="store_true", help="no API calls (smoke test; only BM25 is a real measurement)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, help="only the first N golden questions")
    p.add_argument("--judge-model", default="gpt-6-luna")
    p.add_argument("--min-hit-rate", type=float, help="exit 1 if any retrieval config's Hit@k is below this")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    questions = load_golden()[: args.limit]
    files = corpus_files()
    if args.offline:
        from eval.offline_models import offline_llm, offline_providers

        make_providers, judge = offline_providers, offline_llm("offline-judge")
    else:
        settings = get_settings()
        if not settings.openai_api_key:
            print("OPENAI_API_KEY is not set (use --offline for a smoke test)", file=sys.stderr)
            return 2
        make_providers = lambda: openai_providers(settings)  # noqa: E731
        # The judge endpoint/model may only accept the default temperature.
        judge = OpenAIChatClient(
            args.judge_model,
            api_key=settings.openai_api_key,
            temperature=1.0,
            reasoning_effort="none" if args.judge_model.lower() == "gpt-6-luna" else None,
        )

    clean_workdir()
    stages = ["retrieval", "generation"] if args.stage == "all" else [args.stage]
    retrieval, generation, r_records, g_records = {}, {}, {}, {}
    chunks, models = 0, {}
    for stage in stages:
        names = args.configs or (RETRIEVAL_DEFAULT if stage == "retrieval" else GENERATION_DEFAULT)
        for name in names:
            config = CONFIGS[name]
            providers = make_providers()
            models = {
                "chat": providers.chat.model_name,
                "utility": providers.utility.model_name,
                "embedding": getattr(providers.embedding, "model_name", "?"),
                "judge": judge.model_name,
            }
            print(f"[eval] {stage}: {name} ({config.description})", flush=True)
            app = build_app(config, providers, k=args.k)
            chunks = app.state.vector_store.count()
            if stage == "retrieval":
                r_records[name] = evaluate_retrieval(app, questions, k=args.k)
                retrieval[name] = summarise_retrieval(r_records[name])
                retrieval[name]["valid_offline"] = not (config.needs_embeddings or config.needs_llm)
                s = retrieval[name]
                print(f"       Hit@{args.k} {s['hit']['mean']:.3f}  MRR {s['mrr']['mean']:.3f}")
            else:
                g_records[name] = evaluate_generation(app, questions, judge)
                generation[name] = summarise_generation(g_records[name])
                print(f"       correctness {generation[name]['correctness']['mean']:.3f}  "
                      f"faithfulness {generation[name]['faithfulness']['mean']:.3f}")  # fmt: skip

    meta = {
        "run_id": time.strftime("%Y%m%d-%H%M%S") + ("-offline" if args.offline else ""),
        "offline": args.offline,
        "git_sha": git_sha(),
        "corpus_fingerprint": fingerprint(files),
        "corpus_files": len(files),
        "chunks": chunks,
        "golden_fingerprint": hashlib.sha256(GOLDEN_FILE.read_bytes()).hexdigest()[:12],
        "n_questions": len(questions),
        "k": args.k,
        "models": models,
    }
    out = write_run(meta, r_records, g_records, retrieval, generation)
    print(f"[eval] report: {out / 'summary.md'}")

    if args.min_hit_rate is not None:
        failing = {n: s["hit"]["mean"] for n, s in retrieval.items() if s["hit"]["mean"] < args.min_hit_rate}
        if failing:
            print(f"[eval] FAIL: Hit@{args.k} below {args.min_hit_rate}: {failing}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
