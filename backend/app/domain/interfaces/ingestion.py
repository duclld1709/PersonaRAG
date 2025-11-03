"""Interface for document ingestion."""

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
