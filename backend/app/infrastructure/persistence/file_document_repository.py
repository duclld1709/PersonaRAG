"""File-based document metadata repository."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence
from uuid import UUID

from app.domain.interfaces.document_repository import DocumentRepository
from app.domain.models.document import DocumentRecord


class FileDocumentRepository(DocumentRepository):
    """Persist document metadata as JSON on disk."""

    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self._load()

    def list(self) -> Sequence[DocumentRecord]:
        return sorted(self._records.values(), key=lambda record: record.updated_at, reverse=True)

    def get(self, document_id: UUID) -> DocumentRecord | None:
        return self._records.get(document_id)

    def upsert(self, record: DocumentRecord) -> DocumentRecord:
        record.touch()
        self._records[record.id] = record
        self._persist()
        return record

    def delete(self, document_id: UUID) -> None:
        if document_id in self._records:
            del self._records[document_id]
            self._persist()

    def _load(self) -> dict[UUID, DocumentRecord]:
        if not self._index_path.exists():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        records: dict[UUID, DocumentRecord] = {}
        for item in raw:
            record = DocumentRecord(
                id=UUID(item["id"]),
                filename=item["filename"],
                source_path=Path(item["source_path"]),
                summary=item.get("summary"),
                uploaded_at=datetime.fromisoformat(item["uploaded_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
            )
            records[record.id] = record
        return records

    def _persist(self) -> None:
        data = [
            {
                "id": str(record.id),
                "filename": record.filename,
                "source_path": str(record.source_path),
                "summary": record.summary,
                "uploaded_at": record.uploaded_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
            }
            for record in self._records.values()
        ]
        self._index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
