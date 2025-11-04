"""HuggingFace embeddings adapter."""

from __future__ import annotations

from typing import Iterable, Sequence
import os


from langchain_community.embeddings import HuggingFaceEmbeddings

from app.domain.interfaces.embedding_client import EmbeddingClient


class HuggingFaceEmbeddingClient(EmbeddingClient):
    """Adapter that wraps LangChain's HuggingFaceEmbeddings."""

    def __init__(self, model_name: str, api_token: str | None = None) -> None:
        if api_token:
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", api_token)
        self._client = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: Iterable[str]) -> Sequence[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)
