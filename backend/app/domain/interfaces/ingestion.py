"""Port for document ingestion/indexing used by services.

Defines the minimal contract that an ingestion pipeline must satisfy
so application services do not depend on a concrete implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol
from uuid import UUID


class DocumentIndexer(Protocol):
    """Abstract contract for indexing and managing document chunks."""

    def index_document(
        self,
        *,
        document_id: UUID,
        content: str,
        metadata: Mapping[str, Any] | None = None,
        run_tree: Any | None = None,
    ) -> int: ...

    def delete_document(self, document_id: UUID, *, run_tree: Any | None = None) -> None: ...

    def reset(self) -> None: ...

    def count(self) -> int: ...
