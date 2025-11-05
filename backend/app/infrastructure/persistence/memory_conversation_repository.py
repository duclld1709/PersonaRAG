"""In-memory conversation repository for development."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Sequence
from uuid import UUID

from app.domain.interfaces.conversation_repository import ConversationRepository
from app.domain.models.chat import ChatMessage, Conversation


class InMemoryConversationRepository(ConversationRepository):
    """Retains conversations in process memory."""

    def __init__(self) -> None:
        self._conversations: OrderedDict[UUID, Conversation] = OrderedDict()

    def create(self, title: str | None = None) -> Conversation:
        conversation = Conversation(title=title or "New conversation")
        self._conversations[conversation.id] = conversation
        return conversation

    def get(self, conversation_id: UUID) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def list_recent(self, limit: int) -> Sequence[Conversation]:
        conversations = list(self._conversations.values())
        return list(reversed(conversations))[:limit]

    def append_message(self, conversation_id: UUID, message: ChatMessage) -> Conversation:
        conversation = self._conversations[conversation_id]
        conversation.append(message)
        self._conversations.move_to_end(conversation_id)
        return conversation

    def rename(self, conversation_id: UUID, title: str) -> Conversation:
        conversation = self._conversations[conversation_id]
        conversation.title = title
        conversation.updated_at = datetime.utcnow()
        self._conversations.move_to_end(conversation_id)
        return conversation

    def delete(self, conversation_id: UUID) -> None:
        self._conversations.pop(conversation_id, None)
