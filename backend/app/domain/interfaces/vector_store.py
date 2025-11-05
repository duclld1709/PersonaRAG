"""Abstraction over vector store implementations."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

from langchain_core.documents import Document


class VectorStore(Protocol):
    """Contract for embedding persistence and retrieval."""

    def add_documents(self, documents: Iterable[Document]) -> None: ...

    def delete(self, *, ids: list[str] | None = None, filter: dict | None = None) -> None: ...

    def get(self, *, filter: dict | None = None) -> Sequence[Document]: ...

    def as_retriever(self, **kwargs) -> any:  # type: ignore[valid-type]
        """Return a retriever wrapper from the underlying vector store."""
        ...

    def reset(self) -> None: ...

    def count(self) -> int: ...
