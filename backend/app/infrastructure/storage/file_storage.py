"""Local filesystem storage for uploaded documents."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

from app.domain.interfaces.document_storage import DocumentStorage


class LocalFileStorage(DocumentStorage):
    """Simple file storage adapter for text documents."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path.resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)

    def save_text(self, storage_name: Path | str, content: str) -> Path:
        """Persist raw text content using a storage-safe filename."""

        path = self._resolve(storage_name)
        path.write_text(content, encoding="utf-8")
        return path

    def read_text(self, location: Path | str) -> str:
        """Return text content from disk."""

        path = self._resolve(location)
        return path.read_text(encoding="utf-8")

    def delete(self, location: Path | str) -> None:
        """Remove a stored document if it exists."""

        path = self._resolve(location)
        if path.exists():
            path.unlink()

    def build_text_filename(self, original_filename: str, document_id: UUID) -> str:
        """Derive a deterministic storage filename backed by the document id."""

        stem = Path(original_filename).stem
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "document"
        return f"{safe_stem}_{document_id.hex}.txt"

    def _resolve(self, location: Path | str) -> Path:
        """Resolve a filename or path relative to the storage root."""

        path = location if isinstance(location, Path) else Path(location)
        if not path.is_absolute():
            if path.parts and path.parts[0] == self._base_path.name:
                path = Path(*path.parts[1:])
            path = self._base_path / path
        return path
