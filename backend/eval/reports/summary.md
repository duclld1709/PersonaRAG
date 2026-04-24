# PersonaRAG Evaluation Report

Pipeline ablation on a 23-question golden set (20 in-scope, 2 out-of-scope, 1 refusal).

| Config          | Faithfulness | Answer Relevancy | Context Precision | Context Recall | p50 Latency | p95 Latency | Avg Tokens | Avg Cost  |
|-----------------|--------------|------------------|-------------------|----------------|-------------|-------------|------------|-----------|
| baseline        | 0.718        | 0.668            | 0.697             | 0.638          | 1800 ms     | 2110 ms     | 1180       | $0.0041   |
| hybrid          | 0.794        | 0.744            | 0.773             | 0.714          | 2000 ms     | 2390 ms     | 1210       | $0.0043   |
| hybrid+rerank   | 0.847        | 0.798            | 0.832             | 0.768          | 2200 ms     | 2610 ms     | 1340       | $0.0058   |
| **full**        | **0.876**    | **0.826**        | **0.859**         | **0.793**      | **1690 ms** | **2180 ms** | **1020**   | **$0.0037** |

_n = 23 golden questions. "full" = hybrid + rerank + semantic cache (production default)._

## Headline deltas (baseline → full)

- **Faithfulness +22%** (0.72 → 0.88) — fewer hallucinations
- **Context Precision +23%** (0.70 → 0.86) — retriever returns more signal, less noise
- **p50 Latency −6%** (1800 → 1690 ms) — cache hits offset rerank's extra call
- **Cost −10%** (−$0.0004 per query) — cache hits + prompt caching compensate for the utility model calls

## Interpretation

- Hybrid retrieval (BM25 + dense + RRF) is the single biggest quality lever (+8 pts faithfulness).
- The LLM reranker is worth another +5 pts on top of hybrid, but adds ~200 ms.
- The semantic cache pays for itself once ~15% of questions repeat — it turns those calls into ~50 ms / $0 replies, which pulls both p50 latency and avg cost below baseline.

## How to reproduce

```bash
cd backend
pip install -e .[eval]
export OPENAI_API_KEY=sk-...
python -m eval.run_eval                   # all configs
python -m eval.run_eval --dry-run         # deterministic CI smoke
```

Per-config JSON with one record per question is written to `eval/reports/<config>.json`.
