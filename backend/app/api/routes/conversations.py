"""Conversation endpoints."""

from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_conversation_service
from app.api.schemas.chat import Citation, ConversationDetail, ConversationMessage, UsageSchema
from app.api.schemas.conversations import (
    ConversationCreateRequest,
    ConversationRenameRequest,
    ConversationSummary,
)
from app.domain.models.chat import Conversation
from app.services.conversation_service import ConversationService

router = APIRouter()


@router.get("/", response_model=list[ConversationSummary])
def list_recent_conversations(
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationSummary]:
    conversations = conversation_service.list_recent()
    return [to_summary(convo) for convo in conversations]


@router.post("/", response_model=ConversationDetail, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreateRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetail:
    conversation = conversation_service.create_conversation(title=payload.title)
    return to_detail(conversation)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: UUID,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetail:
    conversation = conversation_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return to_detail(conversation)


@router.patch("/{conversation_id}", response_model=ConversationDetail)
def rename_conversation(
    conversation_id: UUID,
    payload: ConversationRenameRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetail:
    conversation = conversation_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    updated = conversation_service.rename_conversation(conversation_id, payload.title)
    return to_detail(updated)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: UUID,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> None:
    conversation = conversation_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    conversation_service.delete_conversation(conversation_id)


def to_summary(conversation: Conversation) -> ConversationSummary:
    preview = conversation.last_message.content[:120] if conversation.last_message else None
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        last_message_preview=preview,
        updated_at=conversation.updated_at,
    )


def to_detail(conversation: Conversation) -> ConversationDetail:
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        messages=[
            ConversationMessage(
                role=msg.role.value,
                content=msg.content,
                created_at=msg.created_at,
                citations=[Citation(**asdict(c)) for c in msg.citations] if msg.citations else None,
                usage=UsageSchema(**asdict(msg.usage)) if msg.usage else None,
                latency_ms=msg.latency_ms,
                mode=msg.mode,
            )
            for msg in conversation.messages
        ],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )
