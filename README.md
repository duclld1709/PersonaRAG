# PersonaRAG — Research Assistant for Students

> **A measured RAG study assistant on the OpenAI stack.** Upload lecture notes, papers, and textbooks. Ask grounded questions. Get answers with inline citations, or an honest "not in your materials".

<p>
  <a href="https://github.com/duclld1709/PersonaRAG/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/duclld1709/PersonaRAG/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white">
  <img alt="Next.js" src="https://img.shields.io/badge/Next.js-14-black?logo=nextdotjs&logoColor=white">
  <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-gpt--4.1%20%7C%20embeddings-412991?logo=openai&logoColor=white">
  <img alt="Chroma" src="https://img.shields.io/badge/Vector%20DB-Chroma-ff6f61">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## Why this project exists

Students drown in course materials — lecture slides, assigned papers, textbook chapters — and ChatGPT happily hallucinates on top of all of it. **PersonaRAG** is a private research assistant that only answers from the materials _you_ upload, cites every claim, and refuses to help with plagiarism.

It is also a full-stack engineering portfolio project: the usual quality levers of a RAG system (hybrid search, multi-query, reranking, caching, routing) are implemented from scratch, and each is evaluated on a reproducible harness rather than assumed to help.

---

## Evaluation (honest numbers)

> **v2 correction.** The v1 README showed a results table (faithfulness 0.72 → 0.88,
> "−10% cost", and so on). Those numbers came from the harness's `--dry-run` placeholder
> generator, not from a model run. They have been removed. The harness was rebuilt so that
> every number below can be traced to a run in [`backend/eval/reports/runs/`](backend/eval/reports/runs/):
> retrieval [`20261021-223801`](backend/eval/reports/runs/20261021-223801/summary.md),
> answers [`20261021-234059`](backend/eval/reports/runs/20261021-234059/summary.md).

**Setup.** 400 chunks from 2 lecture notes plus 14 sections of *Dive into Deep Learning*
(CC BY-SA, fetched at a pinned commit). 81 golden questions with **verbatim evidence
quotes**, covering single-source, multi-chapter, out-of-scope and academic-integrity cases.
Relevance is "the chunk contains the evidence", not "the chunk comes from the right file".
All means carry 95% bootstrap CIs. Models for these runs: `gpt-6-luna` (chat, utility and
judge, reasoning effort `none`) and `text-embedding-3-large`. Details are in
[`backend/eval/README.md`](backend/eval/README.md).

### Retrieval (k = 5, 75 in-scope questions)

| config | Hit@5 | Recall@5 | MRR | nDCG@5 | p50 latency |
|---|---|---|---|---|---|
| BM25 only | 85.3 [77.3, 93.3] | 80.7 [71.8, 88.9] | 0.707 [0.62, 0.79] | 0.703 [0.62, 0.79] | 4 ms |
| dense (v1-style baseline) | 89.3 [82.7, 96.0] | 86.0 [78.0, 92.7] | 0.702 [0.62, 0.79] | 0.720 [0.64, 0.80] | 333 ms |
| dense + MMR | 77.3 [68.0, 86.7] | 74.0 [64.0, 83.3] | 0.668 [0.57, 0.76] | 0.656 [0.57, 0.75] | 342 ms |
| hybrid (BM25 + dense, RRF) | 89.3 [82.7, 96.0] | 86.0 [78.0, 92.7] | 0.744 [0.66, 0.82] | 0.747 [0.67, 0.82] | 330 ms |
| hybrid + MMR | 77.3 [68.0, 86.7] | 74.0 [64.0, 83.3] | 0.679 [0.59, 0.77] | 0.662 [0.57, 0.75] | 334 ms |
| hybrid + multi-query | 92.0 [85.3, 97.3] | 85.8 [78.4, 92.4] | 0.746 [0.67, 0.83] | 0.746 [0.67, 0.82] | 3.3 s |
| hybrid + rerank | 93.3 [88.0, 98.7] | 89.8 [82.9, 95.3] | 0.839 [0.77, 0.90] | 0.832 [0.76, 0.90] | 2.8 s |
| **full** (hybrid-MMR + multi-query + rerank) | **100.0** [100.0, 100.0] | **95.8** [92.2, 98.7] | **0.898** [0.85, 0.94] | **0.885** [0.84, 0.93] | 7.0 s |

### Answers (end to end through `ChatService`, 81 questions)

| config | faithfulness | citation precision | correctness | context recall | false abstain | OOS ok | refusal ok | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| dense | 95.8 [93.0, 98.2] | 95.2 [92.0, 97.8] | 95.3 [90.7, 98.7] | 86.0 [78.0, 92.7] | 1.3 [0.0, 4.0] | 4/4 | 2/2 | 4.8 / 6.6 s |
| hybrid | 94.7 [91.9, 97.2] | 94.5 [91.6, 97.0] | 96.7 [92.7, 99.3] | 86.7 [78.7, 93.3] | 2.7 [0.0, 6.7] | 4/4 | 2/2 | 5.0 / 6.4 s |
| hybrid + rerank | 95.3 [92.1, 97.7] | 95.2 [92.1, 97.7] | 96.0 [92.0, 98.7] | 89.8 [82.9, 95.3] | 2.7 [0.0, 6.7] | 4/4 | 2/2 | 7.3 / 9.2 s |
| **full** | **98.0** [96.0, 99.4] | **97.0** [94.7, 98.7] | **100.0** [100.0, 100.0] | **95.6** [91.6, 98.7] | **0.0** [0.0, 0.0] | 4/4 | 2/2 | 11.2 / 14.2 s |

**What the numbers say.**

- **The production pipeline (`full`) is the best configuration, and the gap is real.** Against
  the dense baseline its MRR is 0.898 vs 0.702 and its answer correctness 100% vs 95.3%,
  with non-overlapping CIs in both cases. Every in-scope question has evidence in the top 5,
  and no in-scope question was wrongly answered with "not found".
- **The reranker is the biggest single lever on ranking**: hybrid → hybrid + rerank lifts
  MRR from 0.744 to 0.839. Hybrid alone and multi-query alone move the numbers by less than
  their CIs, so on this corpus they are not shown to help in isolation.
- **MMR on its own hurts.** Diversifying the top 5 drops Hit@5 by 12 points (89.3 → 77.3)
  for both dense and hybrid, because it pushes out near-duplicate chunks that carry the
  evidence. It only pays off inside `full`, where multi-query and the reranker recover
  those chunks.
- **Context recall drives answer quality.** Faithfulness and citation precision are high
  (≥ 94.5%) in every configuration. Every correctness error outside `full` falls on the same
  five questions (`cross-02`, `multi-01`, `multi-04`, `d2l-adam-01`, `d2l-shift-03`; three
  of them multi-chapter), and in each case the evidence never reached the prompt (context
  recall 0–33%). `full` retrieves it and answers all five correctly.
- **Quality costs latency.** `full` is 2.3× slower than dense at p50 (11.2 s vs 4.8 s),
  mostly from the multi-query and rerank calls. Streaming hides part of that, but this is
  the trade-off to tune (fewer variations, a smaller rerank pool) if latency matters more.

**Caveats, stated up front.**

- **Same model generates and judges.** `gpt-6-luna` is the chat, utility and judge model,
  so self-preference bias is possible. Judge–human agreement (Cohen's κ) has not been
  measured yet; until it is, treat faithfulness and correctness as upper bounds.
- **Cost was not measured for these runs.** `gpt-6-luna` is not in the pricing table
  (`app/infrastructure/llm/pricing.py`), so only embedding calls were priced and the
  reports show ~$0. Token counts were not stored per question, so cost cannot be
  recovered from the artifacts.
- **Small behaviour sets.** Out-of-scope (4) and refusal (2) cases all pass, but n is too
  small to estimate a rate. The harness counts an out-of-scope question routed to chitchat
  as correct; in `hybrid + rerank` and `full`, "What is the capital of France?" went to
  chitchat and was answered ("Paris") from the model's own knowledge, which that rule is
  lenient about.
- **Models differ from production defaults.** The app ships with `gpt-4.1` / `gpt-4.1-mini`;
  these runs used `gpt-6-luna` for every LLM role.

CI runs the whole harness offline on every PR and **fails if BM25 Hit@5 drops below 0.80**.

```bash
cd backend
pip install -e .[dev]
python -m eval.fetch_corpus
export OPENAI_API_KEY=sk-...
python -m eval.run all                 # retrieval ablation + answers + judge
python -m eval.run all --offline       # no key: BM25 real, the rest is a harness smoke test
```

---

## Engineering audit (v1 → v2)

| v1 problem | Impact | v2 |
|---|---|---|
| README results were dry-run placeholders | Unverifiable claims | Rebuilt harness, evidence-labelled golden set, CIs, run artifacts committed |
| Retrieval "precision/recall" measured by **file name** on a 2-file corpus | Almost any chunk counted as relevant | Evidence-quote relevance on a 400-chunk corpus; Hit/Recall/MRR/nDCG |
| Docstring promised RAGAS; code used word-overlap heuristics | Misleading metrics | Structured LLM judge (claims, citations, correctness) + human-agreement protocol |
| Eval forced every question to "retrieve" and used a second, drifted copy of the pipeline | Router and out-of-scope behaviour never measured | One pipeline: `generate_answer` consumes `stream_answer`; the eval goes through the real router |
| Semantic cache was **global**: ignored mode, conversation and document changes | Quiz answer served for "Explain"; answers citing deleted docs; follow-ups hitting other conversations' answers | Scoped by mode, first turn only, cleared on corpus change, "not found" answers never cached |
| Cost meter counted only the final generation call | Router, rewrite, expansion, rerank and embeddings invisible; cost under-reported | Per-request usage tracker, cost broken down by stage in the API, UI and eval |
| Blocking OpenAI calls inside the async SSE generator | One streaming answer stalled every other request | Pipeline iterated in a thread pool |
| PDF pages joined into one string | Citations never had page numbers | Page boundaries kept; chunks carry `page` |
| Reranker blended LLM score with raw RRF score (~0.01) | The "0.3 × upstream" term was a no-op | RRF rescaled to [0, 1] before blending |
| `dedent()` over an f-string with multi-line excerpts | Every prompt carried stray 8-space indents | Prompt assembled explicitly |
| Cache threshold copied at startup | Settings slider had no effect | Read live from settings |
| `CORS allow_origins=["*"]` with credentials; no upload size limit; raw exception text sent to clients | Security / robustness | Configurable origins, `MAX_UPLOAD_MB` (413), generic stream errors |
| No tests, no CI | Regressions undetected | 46 backend tests (offline fakes), eval gate, frontend type-check/lint/build in CI |

---

## Features

### Retrieval pipeline
- **Hybrid search** — BM25 + dense embeddings, fused via Reciprocal Rank Fusion (configurable weights)
- **Multi-query expansion** — GPT-4.1-mini generates paraphrases to broaden recall
- **LLM cross-encoder reranker** — structured-output scoring blends with upstream evidence
- **Semantic response cache**: cosine similarity on query embeddings, scoped by response mode, first-turn questions only, invalidated when documents change
- **Query router** — classifies retrieve / chitchat / refuse so small-talk doesn't burn retrieval budget
- **Incremental reindex** — per-chunk checksum diff, only re-embed changed chunks

### Generation
- **OpenAI-only stack**: `gpt-4.1` (chat), `gpt-4.1-mini` (router/rewriter/reranker), `text-embedding-3-large`, `omni-moderation-latest`
- **Static system prompt first**, so OpenAI's automatic prompt caching applies (cached tokens are tracked and priced separately)
- **Structured outputs** (`response_format=json_schema`) for query expansion + reranker
- **Citation-first prompting** with explicit refusal fallbacks (_"I could not find this in your materials"_)
- **Six response modes**: Answer, Explain, Summarize, Compare, Outline, Quiz
- **Server-Sent Events** streaming with typed status events (routing → retrieving → reranking → generating)

### Safety
- **OpenAI Moderation API** pre-flight check on every user message
- **Academic integrity guardrail** — refuses plagiarism/AI-detection-evasion requests

### Observability
- **Per-request token & cost accounting across every model call** (router, rewrite, expansion, rerank, embeddings, generation), broken down by stage and persisted to SQLite
- **Usage endpoint** (`/api/v1/usage/summary`) + live cost meter in the UI
- **LangSmith tracing** with graceful degradation when disabled

### Frontend UX (Next.js 14 + Tailwind)
- **Three-pane layout**: Conversation list · Chat · Live Context panel showing retrieved excerpts
- **Inline citation pills** ([1], [2]) that highlight the matching excerpt in the Context panel on hover
- **Thinking indicator** surfaces each pipeline stage while the answer streams
- **Cost meter** in the top bar (last-response latency + total USD + cache badge; hover for per-stage cost)
- **Settings drawer** with sliders for retrieval_k, rerank_top_n, cache threshold, moderation on/off
- **Document library** with tag selector (lecture_notes / paper / textbook / assignment / reference)
- **Dark mode**, drag-and-drop upload, prompt suggestions for the empty state

### Persistence
- **SQLite** (WAL mode) for conversations, documents, and usage events — one connection per thread, `PRAGMA foreign_keys = ON`
- **Chroma** for vector storage with `hnsw:space=cosine`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (Next.js 14 · TypeScript · Tailwind)                  │
│    Chat · Context panel · Library · Settings · Cost meter       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ SSE + REST (/api/v1)
┌──────────────────────────────▼──────────────────────────────────┐
│  FastAPI · Clean Architecture                                   │
│                                                                 │
│  api/   → routes/schemas — thin HTTP layer                      │
│  services/ → business logic (Chat, Conversation, Document)      │
│  retrieval/ → hybrid, rerank, cache, router (NEW)               │
│  domain/ → entities + Protocol interfaces                       │
│  infrastructure/ → OpenAI, Chroma, SQLite, moderation           │
│  ingestion/ → loaders, splitters, diff-based pipeline           │
│  eval/ → golden Q&A set, metrics, ablation runner               │
└─────────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┼────────────────────┐
            ▼                  ▼                    ▼
      ┌──────────┐      ┌────────────┐      ┌──────────────┐
      │  OpenAI  │      │   Chroma   │      │    SQLite    │
      │ gpt-4.1  │      │  vectors   │      │ conversations│
      │ 3-large  │      │  + cosine  │      │ documents    │
      │ mod-api  │      │            │      │ usage_events │
      └──────────┘      └────────────┘      └──────────────┘
```

### Pipeline (one chat turn)

```
user message
   │
   ▼ moderation.check()
   ▼ router.classify()          ─── refuse  ─▶ templated refusal
   │                             └── chitchat ─▶ lightweight reply (utility model)
   ▼ cache.lookup()               ─── hit      ─▶ instant reply  ($0, ~50 ms)
   ▼ rewrite_question()          (utility model, history-aware)
   ▼ expand_query()              (utility model → 3 variations, structured JSON)
   ▼ HybridRetriever             BM25 + vector (MMR) × queries → RRF fusion
   ▼ LLMReranker                 top-N by 0-10 score + evidence blend
   ▼ build_prompt()              system + history + excerpts + mode directive
   ▼ chat.stream()               SSE: status → tokens → citations → usage
   ▼ cache.store()               save answer for future similar queries
   ▼ usage_repository.record()   tokens/cost/latency/cache_hit
```

---

## Tech stack

| Layer          | Stack                                                                |
|----------------|----------------------------------------------------------------------|
| LLM & Embeds   | OpenAI `gpt-4.1`, `gpt-4.1-mini`, `text-embedding-3-large`           |
| Safety         | OpenAI Moderation (`omni-moderation-latest`)                         |
| Retrieval      | rank-bm25 · langchain-chroma · custom RRF + reranker                 |
| Backend        | FastAPI · Uvicorn · Pydantic v2 · SQLite (stdlib)                    |
| Observability  | LangSmith · custom usage repository                                  |
| Frontend       | Next.js 14 · React 18 · TypeScript · Tailwind · SWR · Lucide         |
| Eval           | Evidence-labelled golden set · IR metrics · LLM judge · bootstrap CIs |
| Deploy         | Docker · docker-compose                                              |

---

## Getting started

### 1. Backend

```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate       # PowerShell: .venv\Scripts\Activate.ps1
pip install -e .

cp .env.example .env           # then paste your OPENAI_API_KEY
uvicorn app.main:app --reload
```

Backend runs at [http://localhost:8000/api/v1](http://localhost:8000/api/v1) (OpenAPI docs at `/docs`).

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Frontend runs at [http://localhost:3000](http://localhost:3000).

### 3. Docker (full stack)

```bash
docker compose up --build
```

---

## Project structure

```
PersonaRAG/
├── backend/
│   ├── app/
│   │   ├── api/                  routes · schemas · dependencies
│   │   ├── core/                 config · prompts · logging
│   │   ├── domain/               models · Protocol interfaces
│   │   ├── infrastructure/
│   │   │   ├── llm/              OpenAI chat with token/cost tracking
│   │   │   ├── embeddings/       OpenAI embeddings
│   │   │   ├── moderation/       OpenAI moderation adapter
│   │   │   ├── persistence/      SQLite repositories
│   │   │   ├── storage/          local-disk document storage
│   │   │   └── vectorstores/     Chroma adapter
│   │   ├── retrieval/            bm25 · hybrid · reranker · cache · router
│   │   ├── ingestion/            loaders · splitters · diff-based pipeline
│   │   ├── services/             ChatService · ConversationService · DocumentService · UsageService
│   │   ├── bootstrap.py          dependency graph
│   │   └── main.py               FastAPI entrypoint
│   ├── eval/                     corpus fetcher · evidence-labelled golden set · retrieval + judge metrics · reports/runs/
│   ├── tests/                    46 offline tests (fake embeddings/LLM through the real pipeline)
│   ├── pyproject.toml
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── chat/                 ChatPage (client) + route entry
│   │   ├── components/           Sidebar · Topbar · MessageList · Composer · CitationPanel · DocumentPanel · SettingsDrawer · CostMeter · StatusIndicator · ModeSelect
│   │   ├── hooks/                useTheme · useAutoScroll
│   │   └── lib/                  api client · types · cn helper
│   ├── package.json
│   └── .env.local.example
├── docker-compose.yml
└── LICENSE
```

---

## API surface (selected)

| Method   | Path                                  | Purpose                                    |
|----------|---------------------------------------|--------------------------------------------|
| `POST`   | `/api/v1/chat/stream`                 | SSE chat stream (status + tokens + citations + usage) |
| `POST`   | `/api/v1/chat/respond`                | Non-streaming chat                         |
| `GET`    | `/api/v1/conversations`               | Recent conversations                       |
| `POST`   | `/api/v1/documents`                   | Upload + auto-index (.pdf, .docx, .md, .txt) |
| `GET`    | `/api/v1/documents`                   | List with tags + chunk counts              |
| `GET`    | `/api/v1/settings`                    | Current retrieval/chat settings + option ranges |
| `PATCH`  | `/api/v1/settings`                    | Live-update models, K, rerank top N, cache threshold, moderation |
| `GET`    | `/api/v1/usage/summary`               | Aggregate token/cost/cache-hit/latency    |

---

## Design decisions worth calling out

- **OpenAI-only stack.** The previous iteration supported Gemini + HuggingFace; that optionality cost complexity with no real users asking for it. Stripped.
- **Hybrid over multi-query-cosplay.** The old "hybrid" was just multi-query against a vector store — honest hybrid is BM25 + dense fused by rank, so that's what this is.
- **LLM reranker over Cohere.** Keeps the stack single-vendor and lets the reranker share prompt-caching and billing with the rest of the app.
- **SQLite, not Postgres.** The data model is tiny (conversations + messages + documents + usage). WAL mode handles concurrent reads during streaming; when this outgrows SQLite the `Protocol`-based repositories swap for Postgres without touching services.
- **Evidence-labelled golden set over a framework metric suite.** Every in-scope question carries verbatim evidence quotes, so retrieval is scored deterministically (and gated in CI without an API key); only answer quality needs the LLM judge, whose agreement with human labels should be reported alongside its scores.
- **Clean Architecture was worth it.** Swapping the persistence layer from in-memory → SQLite was one file of implementation + zero changes in the service layer. Interfaces earn their keep here.

---

## Roadmap (not yet in)

- Parent-document retrieval (small chunks for search, big chunks for context)
- Semantic chunking for markdown / code
- Postgres adapter behind the same `Protocol`s
- Per-user auth & document isolation
- Judge–human agreement (Cohen's κ) and a judge from a different model family than the generator
- Price `gpt-6-luna` and store per-question token counts, so the eval reports real cost per stage
- Re-run the answer ablation with the production defaults (`gpt-4.1` / `gpt-4.1-mini`)
- Vietnamese course materials and questions (bilingual embeddings, Vietnamese word segmentation for BM25)

---

## License

MIT — see [LICENSE](./LICENSE).
