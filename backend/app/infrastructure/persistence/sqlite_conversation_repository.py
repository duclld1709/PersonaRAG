"""SQLite-backed conversation repository."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Sequence
from uuid import UUID

from app.domain.interfaces.conversation_repository import ConversationRepository
from app.domain.models.chat import ChatMessage, Citation, Conversation, MessageRole, TokenUsage

from .database import Database


class SqliteConversationRepository(ConversationRepository):
    """Persist conversations and messages to SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, title: str | None = None) -> Conversation:
        conversation = Conversation(title=title or "New conversation")
        with self._db.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (
                    str(conversation.id),
                    conversation.title,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                ),
            )
        return conversation

    def get(self, conversation_id: UUID) -> Conversation | None:
        with self._db.cursor() as cur:
            cur.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (str(conversation_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            messages = self._load_messages(cur, conversation_id)
        return _row_to_conversation(row, messages)

    def list_recent(self, limit: int) -> Sequence[Conversation]:
        with self._db.cursor() as cur:
            cur.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            results: list[Conversation] = []
            for row in rows:
                messages = self._load_messages(cur, UUID(row["id"]))
                results.append(_row_to_conversation(row, messages))
        return results

    def append_message(self, conversation_id: UUID, message: ChatMessage) -> Conversation:
        with self._db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, created_at, citations_json,
                    usage_json, latency_ms, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(conversation_id),
                    message.role.value,
                    message.content,
                    message.created_at.isoformat(),
                    json.dumps([asdict(c) for c in message.citations]) if message.citations else None,
                    json.dumps(asdict(message.usage)) if message.usage else None,
                    message.latency_ms,
                    message.mode,
                ),
            )
            cur.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at.isoformat(), str(conversation_id)),
            )
        return self.get(conversation_id)  # type: ignore[return-value]

    def rename(self, conversation_id: UUID, title: str) -> Conversation:
        now = datetime.utcnow().isoformat()
        with self._db.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, str(conversation_id)),
            )
        return self.get(conversation_id)  # type: ignore[return-value]

    def delete(self, conversation_id: UUID) -> None:
        with self._db.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = ?", (str(conversation_id),))

    def _load_messages(self, cur, conversation_id: UUID) -> list[ChatMessage]:
        cur.execute(
            """
            SELECT role, content, created_at, citations_json, usage_json, latency_ms, mode
              FROM messages WHERE conversation_id = ? ORDER BY id ASC
            """,
            (str(conversation_id),),
        )
        rows = cur.fetchall()
        messages: list[ChatMessage] = []
        for row in rows:
            citations: list[Citation] | None = None
            if row["citations_json"]:
                citations = [Citation(**item) for item in json.loads(row["citations_json"])]
            usage: TokenUsage | None = None
            if row["usage_json"]:
                usage = TokenUsage(**json.loads(row["usage_json"]))
            messages.append(
                ChatMessage(
                    role=MessageRole(row["role"]),
                    content=row["content"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    citations=citations,
                    usage=usage,
                    latency_ms=row["latency_ms"],
                    mode=row["mode"],
                )
            )
        return messages


def _row_to_conversation(row, messages: list[ChatMessage]) -> Conversation:
    return Conversation(
        id=UUID(row["id"]),
        title=row["title"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        messages=messages,
    )
