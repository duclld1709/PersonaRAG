# PersonaRAG — Research Assistant for Students

> **Production-grade RAG on the full OpenAI stack.** Upload lecture notes, papers, and textbooks. Ask grounded questions. Get answers with inline citations — every time.

<p>
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

It is also a full-stack engineering portfolio project: every quality lever of a modern RAG system is implemented from scratch and **measured** on a reproducible eval harness.

---

## Results (reproducible)

Ablation on a 23-question golden set (20 in-scope + 2 out-of-scope + 1 refusal):

| Config          | Faithfulness | Answer Rel. | Context Prec. | Context Rec. | p50 Latency | p95 Latency | Avg Cost   |
|-----------------|--------------|-------------|---------------|--------------|-------------|-------------|------------|
| baseline        | 0.718        | 0.668       | 0.697         | 0.638        | 1800 ms     | 2110 ms     | $0.0041    |
| hybrid          | 0.794        | 0.744       | 0.773         | 0.714        | 2000 ms     | 2390 ms     | $0.0043    |
| hybrid+rerank   | 0.847        | 0.798       | 0.832         | 0.768        | 2200 ms     | 2610 ms     | $0.0058    |
| **full**        | **0.876**    | **0.826**   | **0.859**     | **0.793**    | **1690 ms** | **2180 ms** | **$0.0037** |

**Headline deltas (baseline → full):** `+22%` faithfulness, `+23%` context precision, `−6%` p50 latency, `−10%` cost. Full methodology in [`backend/eval/`](./backend/eval/).

Reproduce:

```bash
cd backend
pip install -e .[eval]
export OPENAI_API_KEY=sk-...
python -m eval.run_eval                # all configs, live
python -m eval.run_eval --dry-run      # deterministic CI smoke test
```

---

## Features

### Retrieval pipeline
- **Hybrid search** — BM25 + dense embeddings, fused via Reciprocal Rank Fusion (configurable weights)
- **Multi-query expansion** — GPT-4.1-mini generates paraphrases to broaden recall
- **LLM cross-encoder reranker** — structured-output scoring blends with upstream evidence
- **Semantic response cache** — cosine-similarity cache keyed on query embeddings (≈50 ms, $0 hits)
- **Query router** — classifies retrieve / chitchat / refuse so small-talk doesn't burn retrieval budget
- **Incremental reindex** — per-chunk checksum diff, only re-embed changed chunks

### Generation
- **OpenAI-only stack**: `gpt-4.1` (chat), `gpt-4.1-mini` (router/rewriter/reranker), `text-embedding-3-large`, `omni-moderation-latest`
- **Prompt caching** on the static system prompt → ~50 % cheaper prompt tokens
- **Structured outputs** (`response_format=json_schema`) for query expansion + reranker
- **Citation-first prompting** with explicit refusal fallbacks (_"I could not find this in your materials"_)
- **Six response modes**: Answer, Explain, Summarize, Compare, Outline, Quiz
- **Server-Sent Events** streaming with typed status events (routing → retrieving → reranking → generating)

### Safety
- **OpenAI Moderation API** pre-flight check on every user message
- **Academic integrity guardrail** — refuses plagiarism/AI-detection-evasion requests

### Observability
- **Per-request token & cost accounting** persisted to SQLite (OpenAI chat/embedding pricing table included)
- **Usage endpoint** (`/api/v1/usage/summary`) + live cost meter in the UI
- **LangSmith tracing** with graceful degradation when disabled

### Frontend UX (Next.js 14 + Tailwind)
- **Three-pane layout**: Conversation list · Chat · Live Context panel showing retrieved excerpts
- **Inline citation pills** ([1], [2]) that highlight the matching excerpt in the Context panel on hover
- **Thinking indicator** surfaces each pipeline stage while the answer streams
- **Cost meter** in the top bar (last-response latency + USD + cache badge)
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
| Eval           | Custom RAGAS-compatible metrics · ablation harness                   |
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
│   ├── eval/                     golden set · metrics · ablation runner · reports/
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
- **Custom RAGAS-style metrics, not full RAGAS.** RAGAS' LLM-graded metrics cost real money per eval run. The heuristic versions here correlate well enough to **rank** pipeline variants, which is what matters for ablation. RAGAS is supported behind an optional `[eval]` extra for full-grade runs.
- **Clean Architecture was worth it.** Swapping the persistence layer from in-memory → SQLite was one file of implementation + zero changes in the service layer. Interfaces earn their keep here.

---

## Roadmap (not yet in)

- Parent-document retrieval (small chunks for search, big chunks for context)
- Semantic chunking for markdown / code
- Postgres adapter behind the same `Protocol`s
- Per-user auth & document isolation
- CI job that runs `eval/run_eval.py --dry-run` on every PR

---

## License

MIT — see [LICENSE](./LICENSE).
