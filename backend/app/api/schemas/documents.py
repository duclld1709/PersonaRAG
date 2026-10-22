"""Document management schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

DocumentTagLiteral = Literal["lecture_notes", "paper", "textbook", "assignment", "reference", "other"]


class DocumentUpsertRequest(BaseModel):
    """Incoming payload for document creation or update."""

    filename: str = Field(min_length=1)
    content: str = Field(min_length=1)
    tag: DocumentTagLiteral = "other"


class DocumentResponse(BaseModel):
    """Serialized document metadata."""

    id: UUID
    filename: str
    summary: str | None = None
    tag: DocumentTagLiteral = "other"
    chunk_count: int = 0
    uploaded_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    """Document metadata plus full content."""

    content: str
