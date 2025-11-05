<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/80dafe91-188a-4c3c-a128-b0656915a676" width="48%" />
  <img src="https://github.com/user-attachments/assets/1c8f1ebc-f331-40b2-9f51-4684072a76c3" width="48%" />
</div>


---

# 🧠 PersonaRAG Apps

**PersonaRAG** is a **Retrieval-Augmented Generation (RAG)** system designed with a **modular layered architecture**, ensuring clear separation between **Presentation**, **Service**, **Domain**, and **Infrastructure** layers.
It combines a **FastAPI backend** and a **Next.js frontend**, offering a scalable, maintainable, and privacy-preserving chat experience powered entirely by **your own uploaded data**.

---

## ✨ Overview

PersonaRAG allows users to upload their personal documents (`.pdf`, `.docx`, `.txt`, etc.) to build a **private knowledge base**.
The chatbot then generates responses **exclusively grounded in the uploaded data**, guaranteeing:

* **Privacy** — No external data sources are used.
* **Contextual accuracy** — Answers come from your own files.
* **User relevance** — Responses adapt to your personal context.

---

## ⚙️ Core Features

* **Streamed responses (SSE)** based on vector-retrieved `.txt` chunks.
* **Conversation history** (limited to the 5 most recent threads).
* **Document CRUD** with **automatic re-indexing** into a Chroma vector store.
* **Hybrid retrieval** using multi-query + **Reciprocal Rank Fusion (RRF)**.
* **Runtime settings UI** to dynamically switch LLMs, embeddings, and parameters.
* **Citations support** — every generated response includes **source references** that trace back to the original document chunks for transparency and verification.
* **Clean modular layering**:

  ```
  API → Services → Domain → Infrastructure → Ingestion
  ```

---

## 🧩 Project Structure

```
PersonaRAG/
├─ backend/
│  ├─ app/
│  │  ├─ api/
│  │  │  ├─ routes/               # FastAPI endpoints
│  │  │  └─ schemas/              # Pydantic request/response models
│  │  ├─ core/                    # Config, logging, settings
│  │  ├─ domain/
│  │  │  ├─ interfaces/           # Repository & client contracts
│  │  │  └─ models/               # Domain entities
│  │  ├─ infrastructure/
│  │  │  ├─ embeddings/           # Embedding providers
│  │  │  ├─ llm/                  # LLM clients (OpenAI, Gemini, etc.)
│  │  │  ├─ persistence/          # Metadata repositories
│  │  │  ├─ storage/              # File storage adapters
│  │  │  └─ vectorstores/         # Chroma / pgvector integrations
│  │  ├─ ingestion/               # Document parsing + indexing logic
│  │  ├─ services/                # Business logic layer
│  │  ├─ main.py                  # FastAPI entrypoint
│  │  └─ bootstrap.py             # Dependency bootstrapper
│  ├─ pyproject.toml
│  └─ README.md
│
├─ frontend/
│  ├─ app/
│  │  ├─ chat/                    # Chat pages
│  │  ├─ components/              # UI components
│  │  ├─ hooks/                   # React hooks
│  │  ├─ lib/                     # API clients & utils
│  │  ├─ globals.css
│  │  ├─ layout.tsx
│  │  └─ page.tsx
│  ├─ public/avatars/
│  ├─ Dockerfile
│  ├─ next.config.mjs
│  ├─ tailwind.config.js
│  ├─ postcss.config.js
│  ├─ tsconfig.json
│  └─ README.md
│
├─ documents/                      # Material during development phases
│
├─ Dockerfile
├─ docker-compose.yml
├─ docker-compose.gpu.yml
├─ README.md
└─ LICENSE
```

---

## 🚀 Getting Started (Local Development)

### Backend Setup

```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate

pip install -e .
cp .env.example .env
uvicorn app.main:app --reload
```

Backend runs at:
👉 **[http://localhost:8000/api/v1](http://localhost:8000/api/v1)**

---

### Frontend Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Frontend runs at:
👉 **[http://localhost:3000](http://localhost:3000)**

Ensure that your `.env.local` file contains:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
```

---

## 🐳 Docker Deployment

### 1️⃣ Install WSL & Ubuntu

```bash
wsl --install -d Ubuntu
```

### 2️⃣ Install NVIDIA Container Toolkit (for GPU usage)

```bash
sudo apt-get update && sudo apt-get install -y curl gnupg2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

### 3️⃣ Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 4️⃣ Build and Run Containers

#### 🧩 Backend Only

**CPU**

```bash
docker build -t personarag-backend:cpu .
docker run --rm -p 8000:8000 --env-file backend/.env personarag-backend:cpu
```

**GPU**

```bash
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  --build-arg TORCH_VARIANT=cu121 \
  -t personarag-backend:gpu .
docker run --gpus all --rm -p 8000:8000 personarag-backend:gpu
```

#### 💬 Frontend Only

```bash
docker build -t personarag-frontend -f frontend/Dockerfile frontend
docker run --rm -p 3000:3000 \
  -e NEXT_PUBLIC_API_BASE_URL=http://host.docker.internal:8000/api/v1 \
  personarag-frontend
```

#### 🧱 Full Stack (Backend + Frontend)

**CPU**

```bash
docker compose up --build
```

**GPU**

```bash
docker compose -f docker-compose.gpu.yml up --build
```

Stop containers:

```bash
docker compose down
```

When using Docker Compose, the frontend automatically communicates with the backend at
👉 `http://backend:8000/api/v1`

---

## 📘 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

