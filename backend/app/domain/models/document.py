"""Domain model for ingested documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class DocumentRecord:
    """Stored document metadata."""

    filename: str
    source_path: Path
    id: UUID = field(default_factory=uuid4)
    summary: Optional[str] = None
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        """Mark the document as updated."""

        self.updated_at = datetime.utcnow()
