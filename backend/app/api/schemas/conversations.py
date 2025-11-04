"""Conversation-related API schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ConversationSummary(BaseModel):
    """Lightweight conversation summary."""

    id: UUID
    title: str
    last_message_preview: str | None = None
    updated_at: datetime


class ConversationCreateRequest(BaseModel):
    """Payload to create a conversation."""

    title: str | None = None


class ConversationRenameRequest(BaseModel):
    """Payload to rename a conversation."""

    title: str
