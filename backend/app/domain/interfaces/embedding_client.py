"""Embedding interface abstraction."""

from typing import Iterable, Protocol, Sequence


class EmbeddingClient(Protocol):
    """Constract for embedding text chunks."""

    def embed_documents(self, texts: Iterable[str]) -> Sequence[list[float]]: ...

    def embed_query(self, text:str) -> list[float]: ...