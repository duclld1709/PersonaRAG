# Frontend (Next.js)

## Prerequisites

- Node.js 18+
- npm or pnpm

## Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
```

Update `.env.local` to point to your FastAPI backend (default `http://localhost:8000/api/v1`).

## Run

```bash
npm run dev
```

The development server starts on `http://localhost:3000`.

## Features

- Conversation sidebar shows the five most recent threads.
- Create a new conversation with the “New” button.
- Chat view streams answers token-by-token using the backend SSE endpoint.
- Document manager lists, previews, uploads, edits, and deletes `.txt` knowledge sources.
- All API calls are managed through `app/lib/api.ts`, keeping UI components lightweight.
