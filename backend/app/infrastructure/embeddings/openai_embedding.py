"""OpenAI embeddings adapter."""

from __future__ import annotations

from typing import Iterable, Sequence

from langchain_openai import OpenAIEmbeddings

from app.domain.interfaces.embedding_client import EmbeddingClient


class OpenAIEmbeddingClient(EmbeddingClient):
    """Adapter that wraps LangChain's OpenAIEmbeddings."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self._client = OpenAIEmbeddings(model=model_name, api_key=api_key)

    def embed_documents(self, texts: Iterable[str]) -> Sequence[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)
