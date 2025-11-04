"""Chat endpoints."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, Iterable

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service, get_conversation_service
from app.api.schemas.chat import ChatResponse, ChatStreamRequest
from app.domain.models.chat import Citation
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService

router = APIRouter()


@router.post("/respond", response_model=ChatResponse)
def chat(
    payload: ChatStreamRequest,
    chat_service: ChatService = Depends(get_chat_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ChatResponse:
    if not conversation_service.get_conversation(payload.conversation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    result = chat_service.generate_answer(payload.conversation_id, payload.question)
    citations_payload = _serialize_citations(result.citations)
    return ChatResponse(answer=result.answer, citations=citations_payload)


@router.post("/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    chat_service: ChatService = Depends(get_chat_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> StreamingResponse:
    if not conversation_service.get_conversation(payload.conversation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    async def event_stream():
        try:
            for event in chat_service.stream_answer(payload.conversation_id, payload.question):
                if not event:
                    await asyncio.sleep(0)
                    continue
                payload_dict = _prepare_stream_payload(event)
                if payload_dict:
                    data = json.dumps(payload_dict)
                    yield f"data: {data}\n\n"
                await asyncio.sleep(0)
        except Exception as exc:  # noqa: BLE001
            error_payload = json.dumps({"error": str(exc)})
            yield f"data: {error_payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _serialize_citations(citations: Iterable[Citation]) -> list[dict[str, Any]]:
    return [asdict(citation) for citation in citations]


def _prepare_stream_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "delta" in event and isinstance(event["delta"], str):
        payload["delta"] = event["delta"]
    if "citations" in event and event["citations"]:
        citations_value = event["citations"]
        if isinstance(citations_value, list) and citations_value and isinstance(citations_value[0], dict):
            payload["citations"] = citations_value
        else:
            payload["citations"] = _serialize_citations(citations_value)
    if "error" in event and isinstance(event["error"], str):
        payload["error"] = event["error"]
    return payload
