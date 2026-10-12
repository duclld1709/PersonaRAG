# PersonaRAG Backend (FastAPI)

OpenAI-only RAG service with hybrid retrieval, LLM reranking, semantic cache, and per-request cost accounting.

## Prerequisites

- Python 3.10+
- An OpenAI API key (`OPENAI_API_KEY`)
- Optional: a LangSmith project for tracing

## Setup

```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate                 # PowerShell: .venv\Scripts\Activate.ps1
pip install -e .                         # runtime only
pip install -e .[dev]                    # + linting/test tooling
cp .env.example .env                     # paste your key
```

## Run

```bash
uvicorn app.main:app --reload
```

- REST + SSE: `http://localhost:8000/api/v1`
- OpenAPI docs: `http://localhost:8000/docs`

## Tests

```bash
pip install -e .[dev]
pytest                                   # ~3 s, no network: offline embeddings/LLM fakes
ruff check . && ruff format --check .
```

## Evaluation

See [`eval/README.md`](eval/README.md). In short:

```bash
python -m eval.fetch_corpus              # once: pinned, checksummed textbook corpus
python -m eval.run all                   # retrieval ablation + end-to-end answers (needs OPENAI_API_KEY)
python -m eval.run all --offline         # no key: BM25 is real, the rest is a harness smoke test
```

## Endpoints

| Method | Path                          | Notes |
|--------|-------------------------------|-------|
| `GET`  | `/api/v1/health/`             | Liveness |
| `GET`  | `/api/v1/conversations`       | Recent conversations |
| `POST` | `/api/v1/conversations`       | Create conversation |
| `GET`  | `/api/v1/conversations/{id}`  | Full history with messages + usage |
| `PATCH`| `/api/v1/conversations/{id}`  | Rename |
| `DELETE`| `/api/v1/conversations/{id}` | Delete |
| `POST` | `/api/v1/chat/stream`         | SSE — status + token + citations + usage |
| `POST` | `/api/v1/chat/respond`        | Non-streaming answer |
| `GET`  | `/api/v1/documents`           | List documents |
| `POST` | `/api/v1/documents`           | Upload .pdf/.docx/.md/.txt with a `tag` |
| `PUT`  | `/api/v1/documents/{id}`      | Replace + reindex |
| `DELETE`| `/api/v1/documents/{id}`     | Delete + purge chunks |
| `GET`  | `/api/v1/settings`            | Current settings + available options |
| `PATCH`| `/api/v1/settings`            | Live-update models, retrieval K, rerank top N, cache threshold, moderation |
| `GET`  | `/api/v1/usage/summary`       | Aggregate usage/cost/latency |
| `GET`  | `/api/v1/usage/recent`        | Most recent N events |

## Module map

```
app/
├── api/
│   ├── routes/        # chat, conversations, documents, settings, usage, health
│   ├── schemas/       # Pydantic request/response models
│   └── dependencies/  # FastAPI DI providers
├── core/
│   ├── config.py      # pydantic-settings (OpenAI-only)
│   ├── prompts.py     # system prompts, mode directives, refusal template
│   └── logging.py
├── domain/
│   ├── models/        # ChatMessage, Citation, Conversation, DocumentRecord, TokenUsage
│   └── interfaces/    # Protocol-based ports (LLMClient, EmbeddingClient, VectorStore, …)
├── infrastructure/
│   ├── llm/           # OpenAI chat/utility client + pricing table
│   ├── embeddings/    # OpenAI embeddings adapter
│   ├── moderation/    # Moderation API wrapper
│   ├── persistence/   # SQLite database + repositories (conversation/document/usage)
│   ├── storage/       # LocalFileStorage
│   └── vectorstores/  # Chroma adapter
├── retrieval/         # BM25Index, HybridRetriever, LLMReranker, SemanticResponseCache, QueryRouter
├── ingestion/         # loaders, splitters, diff-based indexing pipeline
├── services/          # ChatService, ConversationService, DocumentService, UsageService
├── bootstrap.py       # wires everything together
└── main.py            # FastAPI entrypoint
```

## Config reference (`.env`)

See [`.env.example`](./.env.example) for the full list — selected keys:

| Key                          | Default                         | Purpose |
|------------------------------|---------------------------------|---------|
| `OPENAI_API_KEY`             | _(required)_                    | OpenAI credential |
| `OPENAI_CHAT_MODEL`          | `gpt-4.1`                       | Primary generation model |
| `OPENAI_UTILITY_MODEL`       | `gpt-4.1-mini`                  | Router, rewriter, reranker |
| `OPENAI_EMBEDDING_MODEL`     | `text-embedding-3-large`        | Dense retrieval + cache keys |
| `RETRIEVAL_K`                | `5`                             | Vector candidates per query variation |
| `RETRIEVAL_FETCH_K`          | `20`                            | MMR fetch pool |
| `HYBRID_BM25_WEIGHT`         | `0.4`                           | BM25 weight in RRF |
| `HYBRID_VECTOR_WEIGHT`       | `0.6`                           | Dense weight in RRF |
| `RERANK_TOP_N`               | `5`                             | Excerpts after LLM rerank |
| `SEMANTIC_CACHE_THRESHOLD`   | `0.95`                          | Cosine similarity for cache hit |
| `QUERY_VARIATIONS`           | `3`                             | Multi-query expansion count |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150`                 | Splitter |
| `ENABLE_MODERATION`          | `true`                          | Pre-flight safety check |
| `DATABASE_URL`               | `sqlite:///./db/personarag.db`  | SQLite path |
