"""Document management schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUpsertRequest(BaseModel):
    """Incoming payload for document creation or update."""

    filename: str = Field(min_length=1)
    content: str = Field(min_length=1)


class DocumentResponse(BaseModel):
    """Serialized document metadata."""

    id: UUID
    filename: str
    summary: str | None = None
    uploaded_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    """Document metadata plus full content."""

    content: str
