# Backend (FastAPI)

## Prerequisites

- Python 3.10+
- `pip` or a compatible package manager
- Fill API key saved in `.env` from `.env.example`
- If you want to capture detailed traces (retrieval metadata, token usage, costs) in LangSmith, uncomment LangSmith field in `.env` and fill in LangSmith token key.

## Setup

```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

`pyproject.toml` contains the runtime and development dependencies. Installing with `pip install -e .[dev]` will add the optional tooling (linting, tests).

## Running the API

```bash
uvicorn app.main:app --reload
```

Default base URL: `http://localhost:8000/api/v1`.

## Key Endpoints

- `GET /api/v1/health/` – healthcheck
- `GET /api/v1/conversations` – list the five most recent conversations
- `POST /api/v1/conversations` – create a new conversation
- `POST /api/v1/chat/{conversation_id}` – synchronous answer
- `POST /api/v1/chat/{conversation_id}/stream` – Server-Sent Events stream (tokens)
- `GET /api/v1/documents` – list uploaded documents
- `POST /api/v1/documents` – upload `.txt` content
- `PUT /api/v1/documents/{id}` – replace document
- `DELETE /api/v1/documents/{id}` – delete document

## Document Ingestion

Place `.txt` files inside `docs/` at the repository root. The ingestion pipeline runs automatically when the server starts and whenever documents change through the API.

## LangSmith Tracing

**What gets logged**
- `ChatService.stream_answer` and `generate_answer` produce top-level LangSmith runs, including:
  - Retrieval spans with document IDs, chunk indices, and previews for the context that fed the LLM.
  - Citation metadata captured once an answer is generated.
- `DocumentIngestionPipeline.index_document` and `delete_document` create runs for ingestion, showing chunk counts per document.

With tracing enabled, LangChain LLM calls automatically include token usage so you can track cost per request inside LangSmith.
