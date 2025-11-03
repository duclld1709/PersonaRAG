"""Conversation-related application services."""

from __future__ import annotations

import re
from typing import Sequence
from uuid import UUID

from app.core.config import Settings
from app.domain.interfaces.conversation_repository import ConversationRepository
from app.domain.models.chat import ChatMessage, Citation, Conversation, MessageRole


class ConversationService:
    """Coordinates conversation creation and history retrieval."""

    def __init__(self, repository: ConversationRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def create_conversation(self, title: str | None = None) -> Conversation:
        return self._repository.create(title=title)

    def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        return self._repository.get(conversation_id)

    def list_recent(self) -> Sequence[Conversation]:
        return self._repository.list_recent(self._settings.max_conversation_history)

    def add_user_message(self, conversation_id: UUID, content: str) -> Conversation:
        message = ChatMessage(role=MessageRole.USER, content=content)
        conversation = self._repository.append_message(conversation_id, message)

        if self._should_autoname(conversation):
            generated_title = self._generate_title_from_message(content)
            if generated_title:
                conversation = self._repository.rename(conversation_id, generated_title)

        return conversation

    def add_assistant_message(
        self,
        conversation_id: UUID,
        content: str,
        *,
        citations: Sequence[Citation] | None = None,
    ) -> Conversation:
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            citations=list(citations) if citations else None,
        )
        return self._repository.append_message(conversation_id, message)

    def rename_conversation(self, conversation_id: UUID, title: str) -> Conversation:
        return self._repository.rename(conversation_id, title)

    def delete_conversation(self, conversation_id: UUID) -> None:
        self._repository.delete(conversation_id)

    def _should_autoname(self, conversation: Conversation) -> bool:
        """Determine whether the conversation should receive an automatic title."""

        default_title = "new conversation"
        current_title = (conversation.title or "").strip().lower()

        if current_title and current_title != default_title:
            return False

        if not conversation.messages:
            return False

        first_message = conversation.messages[0]
        if first_message.role != MessageRole.USER:
            return False

        return len(conversation.messages) == 1

    def _generate_title_from_message(self, content: str) -> str:
        """Create a short title using the opening words of the user's message."""

        sanitized = re.sub(r"\s+", " ", content or "").strip()
        if not sanitized:
            return "New conversation"

        max_words = 8
        max_length = 60

        words = sanitized.split(" ")
        selected_words = words[:max_words]
        title = " ".join(selected_words)

        if len(title) > max_length:
            title = title[:max_length].rstrip()

        if len(words) > max_words or len(sanitized) > len(title):
            title = title.rstrip(".,;:!-— ") + "…"

        return title
