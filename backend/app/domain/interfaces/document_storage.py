"""Port for document storage operations used by services.

Application services should depend on this protocol instead of any
concrete storage implementation (Local FS, S3, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import UUID


class DocumentStorage(Protocol):
    """Abstract storage contract for raw document contents."""

    def save_text(self, storage_name: Path | str, content: str) -> Path: ...

    def read_text(self, location: Path | str) -> str: ...

    def delete(self, location: Path | str) -> None: ...

    def build_text_filename(self, original_filename: str, document_id: UUID) -> str: ...
