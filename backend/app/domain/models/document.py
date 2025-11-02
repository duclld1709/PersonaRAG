"""Domain model for ingested documents"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional
from datetime import datetime, timezone, timedelta


def now_vn() -> datetime:
    """Return current datetime in Vietnam timezone (UTC+7)."""
    return datetime.now(timezone(timedelta(hours=7)))

@dataclass
class DocumentRecord:
    """Stored document metadata."""

    filename: str
    source_path: Path
    id: UUID = field(default_factory=uuid4)
    summary: Optional[str] = None
    uploaded_at: datetime = field(default_factory=now_vn())
    updated_at: datetime = field(default_factory=now_vn())

    def touch(self) -> None:
        """Mark the document as updated."""

        self.updated_at = now_vn()