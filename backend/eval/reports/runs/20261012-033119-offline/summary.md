# PersonaRAG evaluation: 20261012-033119-offline

> **Smoke test.** Run with `--offline` (hash embeddings, heuristic LLM and judge). Only configurations marked *valid offline* (BM25) are real measurements; every other number only checks the harness.

- commit `9067dd5`, corpus `e838336b837a` (16 files, 400 chunks), golden set `022b102b27d6` (81 questions)
- models: chat `offline-chat`, utility `offline-utility`, embeddings `hash-embeddings`, judge `offline-judge`
- k = 5; values are means with 95% bootstrap CIs

## Retrieval (in-scope questions)

| config | Hit@k | Recall@k | MRR | nDCG@k | p50 latency | cost / query |
|---|---|---|---|---|---|---|
| bm25 *(valid offline)* | 85.3 [77.3, 93.3] | 80.7 [71.8, 88.9] | 0.707 [0.619, 0.790] | 0.703 [0.619, 0.786] | 1 ms | $0.00000 |

