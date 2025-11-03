"""Domain models for chat conversations"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

def _now_vn() -> datetime:
    """Return current datetime in Vietnam timezone (UTC+7)."""
    return datetime.now(timezone(timedelta(hours=7)))

class MessageRole(str, Enum):
    """Permitted chat message roles."""

    USER ="user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Citation:
    """Citation details for retireved context."""

    source: str
    filename: str
    page: Optional[int] = None
    content: str = ""

@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=_now_vn)
    Citations: Optional[List[Citation]] = None

@dataclass
class Conversation:
    """COnversation aggregate root."""

    id: UUID = field(default_factory=uuid4)
    title: str = "New conversation"
    created_at: datetime = field(default_factory=_now_vn)
    updated_at: datetime = field(default_factory=_now_vn)
    messages: List[ChatMessage] = field(default_factory=list)

    def append(self, message: ChatMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = _now_vn()
    
    @property
    def last_message(self) -> Optional[ChatMessage]:
        """Return the most recent message if available."""

        if not self.messages:
            return None
        return self.messages[-1]
    
    def touch(self) -> None:
        self.updated_at = _now_vn()