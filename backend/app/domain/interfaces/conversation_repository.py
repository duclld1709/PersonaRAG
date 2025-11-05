"""Repository abstraction for persisting chat conversations."""

from __future__ import annotations

from typing import Protocol, Sequence
from uuid import UUID

from ..models.chat import ChatMessage, Conversation


class ConversationRepository(Protocol):
    """Conversation repository contract."""

    def create(self, title: str | None = None) -> Conversation: ...

    def get(self, conversation_id: UUID) -> Conversation | None: ...

    def list_recent(self, limit: int) -> Sequence[Conversation]: ...

    def append_message(self, conversation_id: UUID, message: ChatMessage) -> Conversation: ...

    def rename(self, conversation_id: UUID, title: str) -> Conversation: ...

    def delete(self, conversation_id: UUID) -> None: ...
