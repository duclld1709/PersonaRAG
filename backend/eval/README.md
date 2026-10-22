# PersonaRAG evaluation harness

Measures retrieval and answer quality of the **production pipeline** (same ingestion,
retriever, reranker, router and `ChatService` the API uses) on a golden set with
evidence labels.

## Data

| | |
|---|---|
| Corpus | 2 hand-written lecture notes (`corpus/*.md`) + 14 sections of *Dive into Deep Learning* (CC BY-SA 4.0), downloaded by `fetch_corpus.py` at a pinned commit and verified by SHA-256. About 212k characters and 400 chunks, with overlapping topics (overfitting, regularisation, optimisers) that act as realistic distractors. |
| Golden set | `golden_qa.jsonl`, 81 questions: 69 single-source, 6 multi-chapter, 4 out-of-scope, 2 academic-integrity refusals |
| Labels | Every in-scope question lists **verbatim evidence quotes** from the corpus. A test checks that each quote exists. A chunk is relevant if it contains a quote (≥ 60% of it, for quotes split across chunks). |

Relevance is judged by evidence, not by file name. A textbook chunk that states the same
fact as the lecture notes counts as a hit. (v1 scored retrieval by file name on a
two-file corpus, so almost any chunk counted.)

## Metrics

**Retrieval** (`retrieval.py`, in-scope questions, top-k after expansion and reranking):
Hit@k, Recall@k (share of evidence quotes covered, which matters for multi-part questions),
MRR and nDCG@k.

**Answers** (`generation.py` + `judge.py`, end to end through `ChatService`; the router is
*not* bypassed):

| metric | definition |
|---|---|
| faithfulness | claims supported by the excerpts the model saw / all claims (LLM judge) |
| citation precision | cited claims whose `[n]` points at a supporting excerpt / cited claims |
| correctness | judge vs. reference answer: 1 correct, 0.5 partial, 0 incorrect |
| context recall | evidence quotes present in the prompt excerpts |
| false abstention | in-scope questions answered with "I could not find this in your materials" |
| OOS ok / refusal ok | out-of-scope questions abstained, integrity violations refused |
| latency, cost | p50/p95; **total** cost of all model calls per question, broken down by stage |

All means come with 95% bootstrap confidence intervals. With 75 in-scope questions the
intervals are wide (often ±8 points), so differences smaller than that are not evidence
that one configuration is better than another.

## Configurations

| name | retrieval | multi-query | rerank |
|---|---|---|---|
| `bm25` | BM25 | – | – |
| `dense` | embedding similarity (v1-style baseline) | – | – |
| `dense-mmr` | embedding MMR | – | – |
| `hybrid` | BM25 + similarity, RRF | – | – |
| `hybrid-mmr` | BM25 + MMR, RRF | – | – |
| `hybrid+mq` | hybrid | 3 variations | – |
| `hybrid+rerank` | hybrid | – | LLM |
| `full` | BM25 + MMR | 3 variations | LLM (production default) |

The semantic cache and moderation are off during evaluation. Each question is asked
once, so a cache could not hit.

## Running

```bash
python -m eval.fetch_corpus
python -m eval.run retrieval                          # all retrieval configs
python -m eval.run generation --configs dense full    # answers + judge (costs a few dollars)
python -m eval.run all --limit 20                     # quick look
python -m eval.run all --offline                      # no API key (see below)
```

Results go to `reports/runs/<timestamp>/`: `summary.md`, `summary.json`, one JSONL per
configuration with per-question scores and judge output, and `meta.json` (commit, corpus and
golden-set fingerprints, model names).

**Offline mode** swaps in hash embeddings and heuristic models. BM25 makes no model calls,
so its numbers are real measurements, and CI gates on them
(`--configs bm25 --min-hit-rate 0.80`). Every other offline number only shows that the
harness runs, and the report says so.

## Latest results

| run | stage | models |
|---|---|---|
| [`20261021-223801`](reports/runs/20261021-223801/summary.md) | retrieval, all 8 configs | `gpt-6-luna` (utility), `text-embedding-3-large` |
| [`20261021-234059`](reports/runs/20261021-234059/summary.md) | answers: dense, hybrid, hybrid+rerank, full | `gpt-6-luna` (chat, utility, judge) |

Headline: `full` reaches Hit@5 100% [100, 100] and MRR 0.898 [0.85, 0.94] (dense: 89.3% and
0.702), and answer correctness 100% [100, 100] with no false abstentions (dense: 95.3%).
The analysis and caveats are in the [top-level README](../../README.md#evaluation-honest-numbers).

The retrieval run was committed without `summary.md` / `summary.json`; they were rebuilt
from its per-question JSONL with the same `report.summarise_retrieval` and `report.markdown`
the runner uses. The `meta.json` of both runs records commit `719df3b`, but
they ran with the uncommitted `reasoning_effort` change that was committed afterwards in
`b95a216`.

Two gaps in these runs: `gpt-6-luna` has no entry in `app/infrastructure/llm/pricing.py`,
so LLM calls were priced at $0 and the cost columns only cover embeddings; and the same
model generated and judged the answers.

## Judge reliability

An LLM judge has biases (verbosity, self-preference), so judge scores should not be
reported without checking them against people. Recommended protocol: sample around 40
answers from a run, label faithfulness and correctness by hand (blind to the
configuration), and report agreement (Cohen's κ) next to the judge scores. Per-claim judge
verdicts are stored in the generation JSONL for this purpose.

## Known limitations

- 81 questions written by the project author. Real student questions would be messier. Add
  them as the app gets users.
- Evidence matching is lexical. A chunk that paraphrases the answer without containing the
  quote is counted as not relevant, which slightly underestimates retrieval.
- The golden set is English. Vietnamese course material is not covered yet.
