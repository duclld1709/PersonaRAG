# RAG Chatbot Platform

Modular Retrieval-Augmented Generation (RAG) chatbot with a FastAPI backend and a Next.js frontend. The solution follows a layered architecture that isolates presentation, service, domain, and infrastructure logic.

## Structure

- `backend/` – FastAPI application (REST + SSE), document ingestion pipeline, LangChain integrations.
- `frontend/` – Next.js App Router UI with streaming chat and document management.
- `docs/` – Additional documentation (architecture, API overview).

Refer to [ARCHITECTURE.md](ARCHITECTURE.md) for the full directory map and layering overview.

## Getting Started

### Backend (local development)

```bash
cd backend
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate
pip install -e .
cp .env.example .env
uvicorn app.main:app --reload
```

Key environment variables (`backend/.env`):

- `OPENAI_API_KEY` – required for embeddings and chat completion when using OpenAI.
- `GEMINI_API_KEY` – required when choosing Gemini as the LLM provider.
- `HUGGINGFACE_API_TOKEN` – optional, used when switching to Hugging Face embeddings.
- `DOCS_PATH`, `VECTORSTORE_PATH`, `LOGS_PATH` – customise storage locations if needed.

API base URL defaults to `http://localhost:8000/api/v1`. See [docs/backend-api.md](docs/backend-api.md).

### Frontend (local development)

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Ensure `NEXT_PUBLIC_API_BASE_URL` matches the backend URL. The dev server runs at `http://localhost:3000`.

## Docker Usage

Container images are provided for both services. Populate `backend/.env` with the required API keys before building.

```bash
# Backend image alone
docker build -t rag-backend .
docker run --rm -p 8000:8000 --env-file backend/.env rag-backend

# Frontend image alone
docker build -t rag-frontend -f frontend/Dockerfile frontend
docker run --rm -p 3000:3000 \
  -e NEXT_PUBLIC_API_BASE_URL=http://host.docker.internal:8000/api/v1 \
  rag-frontend

# Full stack via Docker Compose
docker compose up --build            # add -d for detached mode
docker compose -f docker-compose.gpu.yml up --build
docker compose down                  # add --volumes to clear docs/db/logs
```

When running with Docker Compose, the frontend automatically talks to the backend over the internal network (`http://backend:8000/api/v1`).

## Core Features

- Streamed chat answers (SSE) grounded in indexed `.txt` documents.
- Conversation management limited to the five most recent threads.
- Document CRUD with automatic re-indexing of the Chroma vector store.
- Multi-query retrieval + reciprocal rank fusion implemented via LangChain.
- Modular layering: API → Services → Domain → Infrastructure → Ingestion.
- Runtime settings UI to switch LLM/embedding providers and retrieval parameters.

## Next Steps

- Add persistence for conversations (e.g., PostgreSQL or MongoDB).
- Extend ingestion CLI for batch operations and schedule indexing jobs.
- Expand automated tests (unit tests for services, integration tests for API/UX).
