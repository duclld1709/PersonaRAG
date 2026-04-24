# PersonaRAG Evaluation Harness

Reproducible offline evaluation of the retrieval and generation pipeline against a
golden Q&A set. Outputs a Markdown report with RAGAS metrics and latency / cost
numbers — the same table that appears in the project README.

## Structure

```
eval/
├── README.md               this file
├── corpus/                 reference documents the golden set is grounded in
├── golden_qa.jsonl         question → expected-answer + ground-truth contexts
├── metrics.py              lightweight RAGAS-compatible metrics (+ heuristics)
├── run_eval.py             entrypoint: run all pipeline configs on the golden set
└── reports/                written by run_eval.py
```

## Running

```bash
cd backend
pip install -e .[eval]
python -m eval.run_eval --configs baseline hybrid rerank+cache
```

Requires `OPENAI_API_KEY` in the environment. Each config writes
`reports/<config>.json` with the per-question results and aggregates.

The `run_eval.py` script can also be executed in **dry-run mode**
(`--dry-run`) which skips all OpenAI calls and produces deterministic
placeholder numbers — useful for CI smoke testing.

## Metrics

We report four metrics per config:

| Metric               | Description                                                    |
|----------------------|----------------------------------------------------------------|
| `faithfulness`       | Fraction of answer claims entailed by the retrieved excerpts   |
| `answer_relevancy`   | Embedding similarity between the question and the answer       |
| `context_precision`  | Share of retrieved chunks that were actually relevant          |
| `context_recall`     | Share of ground-truth chunks that appeared in retrieval        |

Plus operational numbers: p50/p95 latency, tokens per query, and cost per query.
