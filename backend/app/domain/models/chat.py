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


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    citations: Optional[List[Citation]] = None


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
