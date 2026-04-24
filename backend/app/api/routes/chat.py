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
from app.core.prompts import ChatMode
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
    result = chat_service.generate_answer(
        payload.conversation_id, payload.question, mode=ChatMode(payload.mode)
    )
    return ChatResponse(
        answer=result.answer,
        citations=[_citation_to_dict(c) for c in result.citations],  # type: ignore[list-item]
        usage=asdict(result.usage),  # type: ignore[arg-type]
        latency_ms=result.latency_ms,
        mode=result.mode,
        cache_hit=result.cache_hit,
        refused=result.refused,
    )


@router.post("/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    chat_service: ChatService = Depends(get_chat_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> StreamingResponse:
    if not conversation_service.get_conversation(payload.conversation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    mode = ChatMode(payload.mode)

    async def event_stream():
        try:
            for event in chat_service.stream_answer(
                payload.conversation_id, payload.question, mode=mode
            ):
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


def _citation_to_dict(c: Citation) -> dict[str, Any]:
    return asdict(c)


def _serialize_citations(citations: Iterable[Citation]) -> list[dict[str, Any]]:
    return [asdict(c) for c in citations]


_STATUS_ALLOWED = {"routing", "retrieving", "reranking", "generating", "cache_hit", "chitchat", "refused"}
_PASSTHROUGH_KEYS = {"usage", "latency_ms", "cache_hit", "mode", "intent"}


def _prepare_stream_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "status" in event and event["status"] in _STATUS_ALLOWED:
        payload["status"] = event["status"]
    if "delta" in event and isinstance(event["delta"], str):
        payload["delta"] = event["delta"]
    if "citations" in event and event["citations"]:
        value = event["citations"]
        if isinstance(value, list) and value and isinstance(value[0], dict):
            payload["citations"] = value
        else:
            payload["citations"] = _serialize_citations(value)
    if "error" in event and isinstance(event["error"], str):
        payload["error"] = event["error"]
    for key in _PASSTHROUGH_KEYS:
        if key in event and event[key] is not None:
            payload[key] = event[key]
    return payload
