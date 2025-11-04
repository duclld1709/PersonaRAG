# RAG Chatbot Project Architecture

This project implements a Retrieval-Augmented Generation (RAG) chatbot with a FastAPI backend and a Next.js frontend. The solution follows a modular layered architecture to keep concerns isolated and to simplify future extensions.

## High-Level Overview

- `backend/`: FastAPI application exposing REST and streaming endpoints for chat conversations and document management.
- `frontend/`: Next.js web application that consumes the backend, renders chat history, and streams model responses to the UI.
- `docs/`: Additional documentation, including API contracts and deployment notes.

## Backend Layering

```
backend/
├── app/
│   ├── api/                 # FastAPI routers (presentation layer)
│   │   ├── dependencies/    # Shared dependency providers
│   │   ├── routes/          # Versioned route modules
│   │   └── schemas/         # Pydantic request/response models
│   ├── core/                # Application-level configuration and utilities
│   ├── domain/              # Domain models and interfaces
│   ├── services/            # Use cases orchestrating domain logic
│   ├── infrastructure/      # Adapters for external systems (vector DB, LLM, storage)
│   ├── ingestion/           # Document ingestion pipeline (split, embed, persist)
│   └── main.py              # FastAPI entrypoint
│
├── tests/                   # Backend tests
└── pyproject.toml           # Project configuration and dependencies
```

### Layer Responsibilities

- **API (Presentation)**: Request validation, routing, SSE support, HTTP concerns.
- **Service (Application)**: Orchestrates workflows like chat turns, conversation history, and document CRUD using domain interfaces.
- **Domain**: Core business logic and abstract interfaces, free of frameworks.
- **Infrastructure**: Concrete implementations for the domain interfaces (Chroma vector store, OpenAI, file storage).
- **Ingestion**: Batch utilities for document loading, splitting, and embedding aligned with infrastructure abstractions.
- **Core**: Settings, logging, dependency injection helpers.

### Ingestion and Retrieval Flow

1. `ingestion/loaders.py` loads `.txt` documents from a configurable directory.
2. `ingestion/splitters.py` produces chunks using recursive splitting strategies.
3. `services/document_service.py` coordinates ingestion, calling infrastructure adapters to persist metadata and vectors.
4. `services/chat_service.py` orchestrates retrieval (multi-query, reciprocal rank fusion) and response generation with streaming support.

### Streaming Responses

- Chat endpoint exposes a Server-Sent Events (SSE) stream.
- `services/chat_service.py` yields tokenized responses from the LLM adapter.
- `api/routes/chat.py` packages tokens as SSE messages for the frontend.

## Frontend Layering

```
frontend/
├── app/                      # Next.js App Router
│   ├── (chat)/               # Chat feature routes
│   ├── components/           # Reusable UI components (ChatWindow, MessageList, etc.)
│   ├── hooks/                # Data fetching and SSE hooks
│   └── lib/                  # API client utilities and shared types
├── public/                   # Static assets
├── styles/                   # Global styling
└── package.json              # Dependencies and scripts
```

### Frontend Features

- Show the five most recent conversations with lazy loading for older sessions.
- Create a new conversation and load conversation history.
- Upload, list, update, or delete text documents (uses backend REST endpoints).
- Post questions and stream responses token-by-token using EventSource or WebSocket.
- Type-safe API layer using TypeScript definitions that mirror backend schemas.

## Shared Contracts

- JSON schemas are defined in `backend/app/api/schemas`. Equivalent TypeScript types live in `frontend/app/lib/types`.
- Document metadata contains `id`, `filename`, `content_summary`, `uploaded_at`, and status flags.
- Chat messages track `conversation_id`, `role`, `content`, and `created_at`.

## Environment and Configuration

- Environment variables stored in `.env` (backend) and `.env.local` (frontend).
- The backend uses `pydantic-settings` for configuration and integrates with `langchain` components through adapters.
- The frontend reads `NEXT_PUBLIC_API_BASE_URL` to communicate with the backend.

## Development Workflow

1. Start the backend (FastAPI with Uvicorn).
2. Run the ingestion CLI to index/update documents.
3. Launch the Next.js dev server.
4. Interact with the chat UI and manage documents.

## Next Steps

- Scaffold backend and frontend codebases following this structure.
- Implement automated tests (unit for services, integration for API routes).
- Containerize the solution with Docker Compose for local dev parity.

