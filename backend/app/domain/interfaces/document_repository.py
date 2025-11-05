"""Repository abstraction for document metadata and storage."""

from __future__ import annotations

from typing import Protocol, Sequence
from uuid import UUID

from ..models.document import DocumentRecord


class DocumentRepository(Protocol):
    """Document repository contract."""

    def list(self) -> Sequence[DocumentRecord]: ...

    def get(self, document_id: UUID) -> DocumentRecord | None: ...

    def upsert(self, record: DocumentRecord) -> DocumentRecord: ...

    def delete(self, document_id: UUID) -> None: ...
