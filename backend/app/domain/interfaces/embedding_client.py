"""Embedding interface abstraction."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence


class EmbeddingClient(Protocol):
    """Contract for embedding text chunks."""

    def embed_documents(self, texts: Iterable[str]) -> Sequence[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
