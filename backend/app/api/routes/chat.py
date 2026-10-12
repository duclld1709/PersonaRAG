"""Chat endpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Iterable

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.api.dependencies import get_chat_service, get_conversation_service
from app.api.schemas.chat import ChatResponse, ChatStreamRequest
from app.core.prompts import ChatMode
from app.domain.models.chat import Citation
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/respond", response_model=ChatResponse)
def chat(
    payload: ChatStreamRequest,
    chat_service: ChatService = Depends(get_chat_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ChatResponse:
    if not conversation_service.get_conversation(payload.conversation_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    result = chat_service.generate_answer(payload.conversation_id, payload.question, mode=ChatMode(payload.mode))
    return ChatResponse(
        answer=result.answer,
        citations=[_citation_to_dict(c) for c in result.citations],  # type: ignore[list-item]
        usage=asdict(result.usage),  # type: ignore[arg-type]
        latency_ms=result.latency_ms,
        mode=result.mode,
        cache_hit=result.cache_hit,
        refused=result.refused,
        intent=result.intent,
        cost_breakdown=result.cost_breakdown,
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
        # The pipeline makes blocking OpenAI/Chroma calls. v1 ran it directly in
        # the event loop, so one streaming answer stalled every other request.
        events = iterate_in_threadpool(
            iter(chat_service.stream_answer(payload.conversation_id, payload.question, mode=mode))
        )
        try:
            async for event in events:
                payload_dict = _prepare_stream_payload(event)
                if payload_dict:
                    yield f"data: {json.dumps(payload_dict)}\n\n"
        except Exception:  # noqa: BLE001
            logger.exception("chat stream failed")
            yield f"data: {json.dumps({'error': 'The assistant failed to answer. Please try again.'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _citation_to_dict(c: Citation) -> dict[str, Any]:
    return asdict(c)


def _serialize_citations(citations: Iterable[Citation]) -> list[dict[str, Any]]:
    return [asdict(c) for c in citations]


_STATUS_ALLOWED = {"routing", "retrieving", "reranking", "generating", "cache_hit", "chitchat", "refused"}
_PASSTHROUGH_KEYS = {"usage", "latency_ms", "cache_hit", "mode", "intent", "refused", "cost_breakdown"}


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
