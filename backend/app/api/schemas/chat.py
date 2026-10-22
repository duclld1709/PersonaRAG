"""Pydantic schemas related to chat endpoints."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ChatModeLiteral = Literal["answer", "explain", "summarize", "compare", "outline", "quiz"]


class ChatRequest(BaseModel):
    """Incoming chat question."""

    question: str = Field(min_length=1)
    mode: ChatModeLiteral = "answer"


class ChatStreamRequest(ChatRequest):
    """Streaming chat payload including conversation context."""

    conversation_id: UUID


class Citation(BaseModel):
    """Citation payload returned alongside answers."""

    source: str
    filename: str
    page: int | None = None
    content: str
    document_id: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    tag: str | None = None


class UsageSchema(BaseModel):
    """Token and cost accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: UsageSchema = Field(default_factory=UsageSchema)
    latency_ms: int = 0
    mode: str = "answer"
    cache_hit: bool = False
    refused: bool = False
    intent: str = "retrieve"
    cost_breakdown: dict[str, float] = Field(default_factory=dict, description="USD per pipeline stage")


class ConversationMessage(BaseModel):
    """Serialized chat message."""

    role: str
    content: str
    created_at: datetime
    citations: list[Citation] | None = None
    usage: UsageSchema | None = None
    latency_ms: int | None = None
    mode: str | None = None


class ConversationDetail(BaseModel):
    """Conversation detail with message history."""

    id: UUID
    title: str
    messages: list[ConversationMessage]
    created_at: datetime
    updated_at: datetime
