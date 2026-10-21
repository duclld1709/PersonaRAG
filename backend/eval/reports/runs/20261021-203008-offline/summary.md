# PersonaRAG evaluation: 20261021-203008-offline

> **Smoke test.** Run with `--offline` (hash embeddings, heuristic LLM and judge). Only configurations marked *valid offline* (BM25) are real measurements; every other number only checks the harness.

- commit `719df3b`, corpus `2ab19c3bc680` (16 files, 400 chunks), golden set `a15dda83e0f4` (81 questions)
- models: chat `offline-chat`, utility `offline-utility`, embeddings `hash-embeddings`, judge `offline-judge`
- k = 5; values are means with 95% bootstrap CIs

## Retrieval (in-scope questions)

| config | Hit@k | Recall@k | MRR | nDCG@k | p50 latency | cost / query |
|---|---|---|---|---|---|---|
| bm25 *(valid offline)* | 85.3 [77.3, 93.3] | 80.7 [71.8, 88.9] | 0.707 [0.619, 0.790] | 0.703 [0.619, 0.786] | 1 ms | $0.00000 |
| dense | 52.0 [40.0, 62.7] | 50.2 [39.3, 61.3] | 0.346 [0.250, 0.439] | 0.370 [0.279, 0.461] | 2 ms | $0.00000 |
| dense-mmr | 38.7 [28.0, 49.3] | 35.8 [25.6, 46.0] | 0.304 [0.206, 0.396] | 0.297 [0.207, 0.388] | 3 ms | $0.00000 |
| hybrid | 52.0 [40.0, 62.7] | 50.2 [39.3, 61.3] | 0.446 [0.340, 0.550] | 0.442 [0.343, 0.545] | 3 ms | $0.00000 |
| hybrid-mmr | 38.7 [28.0, 49.3] | 35.8 [25.6, 46.0] | 0.363 [0.259, 0.473] | 0.338 [0.241, 0.440] | 4 ms | $0.00000 |
| hybrid+mq | 52.0 [40.0, 62.7] | 50.2 [39.3, 61.3] | 0.446 [0.340, 0.550] | 0.442 [0.343, 0.545] | 3 ms | $0.00100 |
| hybrid+rerank | 68.0 [57.3, 78.7] | 63.3 [53.1, 73.6] | 0.460 [0.367, 0.556] | 0.480 [0.389, 0.570] | 10 ms | $0.00100 |
| full | 65.3 [54.7, 76.0] | 60.0 [49.3, 70.4] | 0.424 [0.334, 0.519] | 0.442 [0.355, 0.530] | 17 ms | $0.00200 |

## Answers (end to end)

| config | faithfulness | citation precision | correctness | context recall | false abstain | OOS ok | refusal ok | p50 / p95 latency | cost / question |
|---|---|---|---|---|---|---|---|---|---|
| dense | 40.7 [36.0, 44.7] | 0.0 [0.0, 0.0] | 10.0 [4.0, 16.7] | 50.2 [39.3, 61.3] | 18.7 [10.7, 28.0] | 75.0 [25.0, 100.0] | 100.0 [100.0, 100.0] | 20 / 21 ms | $0.0020 |
| hybrid | 40.0 [35.3, 44.7] | 0.0 [0.0, 0.0] | 10.7 [4.7, 17.3] | 50.2 [39.3, 61.3] | 18.7 [10.7, 28.0] | 75.0 [25.0, 100.0] | 100.0 [100.0, 100.0] | 21 / 23 ms | $0.0020 |
| hybrid+rerank | 43.3 [39.3, 46.7] | 0.0 [0.0, 0.0] | 11.3 [6.0, 18.0] | 63.3 [53.1, 73.6] | 9.3 [2.7, 16.0] | 50.0 [0.0, 100.0] | 100.0 [100.0, 100.0] | 22 / 25 ms | $0.0029 |
| full | 43.3 [39.3, 46.7] | 0.0 [0.0, 0.0] | 12.7 [6.7, 20.0] | 60.0 [49.3, 70.4] | 9.3 [2.7, 16.0] | 50.0 [0.0, 100.0] | 100.0 [100.0, 100.0] | 26 / 29 ms | $0.0039 |

Mean cost per question by pipeline stage (USD):

- **dense**: generate 0.00096, retrieve 0.00000, router 0.00100 (judge total $0.075)
- **hybrid**: generate 0.00096, retrieve 0.00000, router 0.00100 (judge total $0.075)
- **hybrid+rerank**: generate 0.00096, rerank 0.00096, retrieve 0.00000, router 0.00100 (judge total $0.075)
- **full**: expand 0.00096, generate 0.00096, rerank 0.00096, retrieve 0.00000, router 0.00100 (judge total $0.075)
