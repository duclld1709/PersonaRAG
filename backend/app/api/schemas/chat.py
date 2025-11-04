"""Pydantic schemas related to chat endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat question."""

    question: str = Field(min_length=1)


class ChatStreamRequest(ChatRequest):
    """Streaming chat payload including conversation context."""

    conversation_id: UUID


class Citation(BaseModel):
    """Citation payload returned alongside answers."""

    source: str
    filename: str
    page: int | None = None
    content: str


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    """Serialized chat message."""

    role: str
    content: str
    created_at: datetime
    citations: list[Citation] | None = None


class ConversationDetail(BaseModel):
    """Conversation detail with message history."""

    id: UUID
    title: str
    messages: list[ConversationMessage]
    created_at: datetime
    updated_at: datetime
