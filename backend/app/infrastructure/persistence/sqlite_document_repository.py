"""SQLite-backed document repository."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence
from uuid import UUID

from app.domain.interfaces.document_repository import DocumentRepository
from app.domain.models.document import DocumentRecord, DocumentTag

from .database import Database


class SqliteDocumentRepository(DocumentRepository):
    """Persist document metadata to SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def list(self) -> Sequence[DocumentRecord]:
        with self._db.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, source_path, summary, tag, chunk_count, uploaded_at, updated_at
                  FROM documents ORDER BY updated_at DESC
                """
            )
            rows = cur.fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, document_id: UUID) -> DocumentRecord | None:
        with self._db.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, source_path, summary, tag, chunk_count, uploaded_at, updated_at
                  FROM documents WHERE id = ?
                """,
                (str(document_id),),
            )
            row = cur.fetchone()
        return _row_to_record(row) if row else None

    def upsert(self, record: DocumentRecord) -> DocumentRecord:
        record.touch()
        with self._db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, filename, source_path, summary, tag, chunk_count, uploaded_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    source_path = excluded.source_path,
                    summary = excluded.summary,
                    tag = excluded.tag,
                    chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
                """,
                (
                    str(record.id),
                    record.filename,
                    str(record.source_path),
                    record.summary,
                    record.tag.value,
                    record.chunk_count,
                    record.uploaded_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def delete(self, document_id: UUID) -> None:
        with self._db.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = ?", (str(document_id),))


def _row_to_record(row) -> DocumentRecord:
    return DocumentRecord(
        id=UUID(row["id"]),
        filename=row["filename"],
        source_path=Path(row["source_path"]),
        summary=row["summary"],
        tag=DocumentTag(row["tag"]),
        chunk_count=row["chunk_count"],
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
