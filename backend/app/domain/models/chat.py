"""Domain models for chat conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4


class MessageRole(str, Enum):
    """Permitted chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Citation:
    """Citation details for retrieved context."""

    source: str
    filename: str
    page: Optional[int] = None
    content: str = ""
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    tag: Optional[str] = None


@dataclass
class TokenUsage:
    """Token and cost accounting for a single LLM call or pipeline run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: TokenUsage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.cost_usd = round(self.cost_usd + other.cost_usd, 6)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    citations: Optional[List[Citation]] = None
    usage: Optional[TokenUsage] = None
    latency_ms: Optional[int] = None
    mode: Optional[str] = None


@dataclass
class Conversation:
    """Conversation aggregate root."""

    id: UUID = field(default_factory=uuid4)
    title: str = "New conversation"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    messages: List[ChatMessage] = field(default_factory=list)

    def append(self, message: ChatMessage) -> None:
        """Add a message to the conversation."""

        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    @property
    def last_message(self) -> Optional[ChatMessage]:
        """Return the most recent message if available."""

        if not self.messages:
            return None
        return self.messages[-1]
