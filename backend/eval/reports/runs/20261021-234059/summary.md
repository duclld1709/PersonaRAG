# PersonaRAG evaluation: 20261021-234059

- commit `719df3b`, corpus `2ab19c3bc680` (16 files, 400 chunks), golden set `a15dda83e0f4` (81 questions)
- models: chat `gpt-6-luna`, utility `gpt-6-luna`, embeddings `text-embedding-3-large`, judge `gpt-6-luna`
- k = 5; values are means with 95% bootstrap CIs

## Answers (end to end)

| config | faithfulness | citation precision | correctness | context recall | false abstain | OOS ok | refusal ok | p50 / p95 latency | cost / question |
|---|---|---|---|---|---|---|---|---|---|
| dense | 95.8 [93.0, 98.2] | 95.2 [92.0, 97.8] | 95.3 [90.7, 98.7] | 86.0 [78.0, 92.7] | 1.3 [0.0, 4.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 4830 / 6585 ms | $0.0000 |
| hybrid | 94.7 [91.9, 97.2] | 94.5 [91.6, 97.0] | 96.7 [92.7, 99.3] | 86.7 [78.7, 93.3] | 2.7 [0.0, 6.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 4987 / 6407 ms | $0.0000 |
| hybrid+rerank | 95.3 [92.1, 97.7] | 95.2 [92.1, 97.7] | 96.0 [92.0, 98.7] | 89.8 [82.9, 95.3] | 2.7 [0.0, 6.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 7255 / 9239 ms | $0.0000 |
| full | 98.0 [96.0, 99.4] | 97.0 [94.7, 98.7] | 100.0 [100.0, 100.0] | 95.6 [91.6, 98.7] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 11202 / 14151 ms | $0.0000 |

Mean cost per question by pipeline stage (USD):

- **dense**: generate 0.00000, retrieve 0.00000, router 0.00000 (judge total $0.000)
- **hybrid**: generate 0.00000, retrieve 0.00000, router 0.00000 (judge total $0.000)
- **hybrid+rerank**: chitchat 0.00000, generate 0.00000, rerank 0.00000, retrieve 0.00000, router 0.00000 (judge total $0.000)
- **full**: chitchat 0.00000, expand 0.00000, generate 0.00000, rerank 0.00000, retrieve 0.00001, router 0.00000 (judge total $0.000)
