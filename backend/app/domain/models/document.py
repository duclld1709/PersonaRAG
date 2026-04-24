"""Domain model for ingested documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4


class DocumentTag(str, Enum):
    """Student-oriented document categories driving retrieval filters and UI."""

    LECTURE_NOTES = "lecture_notes"
    PAPER = "paper"
    TEXTBOOK = "textbook"
    ASSIGNMENT = "assignment"
    REFERENCE = "reference"
    OTHER = "other"


@dataclass
class DocumentRecord:
    """Stored document metadata."""

    filename: str
    source_path: Path
    id: UUID = field(default_factory=uuid4)
    summary: Optional[str] = None
    tag: DocumentTag = DocumentTag.OTHER
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    chunk_count: int = 0

    def touch(self) -> None:
        """Mark the document as updated."""

        self.updated_at = datetime.utcnow()
