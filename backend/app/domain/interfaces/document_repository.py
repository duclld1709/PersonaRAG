"""Repository abstraction for document metadata and storage."""

from typing import Protocol, Sequence
from uuid import UUID

from ..models.document import DocumentRecord


class DocumentRepository(Protocol):
    """Document repository contract."""

    def list(self) -> Sequence[DocumentRecord]: ...

    def get(self, document_id: UUID) -> DocumentRecord | None: ...

    def upsert(self, recor: DocumentRecord) -> DocumentRecord: ...

    def delete(self, document_id: UUID) -> None: ...

    