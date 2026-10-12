from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.bootstrap import Providers
from app.core.config import Settings
from app.infrastructure.offline import HashEmbeddings, KeywordModeration
from app.main import create_application
from eval.offline_models import offline_llm

NOTES = """# Overfitting

Overfitting occurs when a model learns the training data too well, including the noise, so it
fails to generalise to unseen examples. Remedies include regularisation, early stopping and dropout.

# Cross-validation

k-fold cross-validation partitions the training data into k folds and averages the held-out error.
"""


def make_settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings(
        DATABASE_URL=f"sqlite:///{tmp_path / 'test.db'}",
        OPENAI_API_KEY="sk-test",
        ENABLE_MODERATION=True,
        QUERY_VARIATIONS=2,
        RETRIEVAL_K=3,
        RERANK_TOP_N=2,  # fewer than the 3 chunks of NOTES, so the reranker actually runs
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=20,
        CORS_ORIGINS=["http://localhost:3000"],
        **overrides,
    )
    settings.docs_path = tmp_path / "docs"
    settings.vectorstore_path = tmp_path / "chroma"
    settings.logs_path = tmp_path / "logs"
    settings.docs_path.mkdir(exist_ok=True)
    return settings


@pytest.fixture
def providers() -> Providers:
    # Heuristic offline models: router, rewrite, expansion, rerank and answers all run
    # through the real pipeline code with deterministic outputs and token usage.
    return Providers(
        embedding=HashEmbeddings(),
        chat=offline_llm("chat-model"),
        utility=offline_llm("utility-model"),
        moderation=KeywordModeration(),
    )


@pytest.fixture
def app(tmp_path, providers):
    return create_application(make_settings(tmp_path), providers=providers)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def indexed_client(client):
    res = client.post("/api/v1/documents/", files={"file": ("notes.md", NOTES.encode(), "text/markdown")})
    assert res.status_code == 201, res.text
    return client


def new_conversation(client) -> str:
    return client.post("/api/v1/conversations/", json={}).json()["id"]


def stream(client, conversation_id: str, question: str, mode: str = "answer") -> list[dict]:
    """POST /chat/stream and parse the SSE events."""

    res = client.post(
        "/api/v1/chat/stream", json={"conversation_id": conversation_id, "question": question, "mode": mode}
    )
    assert res.status_code == 200
    events = []
    for line in res.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line.removeprefix("data: ")))
    assert res.text.rstrip().endswith("data: [DONE]")
    return events
