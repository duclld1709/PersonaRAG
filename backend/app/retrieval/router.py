"""Cheap query router that chooses retrieve vs chitchat vs refuse."""

from __future__ import annotations

from enum import Enum
from logging import getLogger

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.prompts import ROUTER_SYSTEM_PROMPT
from app.domain.interfaces.llm_client import LLMClient

logger = getLogger(__name__)


class QueryIntent(str, Enum):
    """Classification buckets returned by the router."""

    RETRIEVE = "retrieve"
    CHITCHAT = "chitchat"
    REFUSE = "refuse"


_VALID = {intent.value for intent in QueryIntent}


class QueryRouter:
    """Wraps a utility model to classify the student's latest message.

    Falls back to `RETRIEVE` if the model errors out, so observability issues
    never silently drop a student's question.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def classify(self, message: str) -> QueryIntent:
        trimmed = message.strip()
        if not trimmed:
            return QueryIntent.CHITCHAT
        try:
            response = self._client.invoke([SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=trimmed)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Router call failed, defaulting to retrieve: %s", exc)
            return QueryIntent.RETRIEVE
        raw = (response.content or "").strip().lower()
        token = raw.split()[0] if raw else ""
        if token in _VALID:
            return QueryIntent(token)
        return QueryIntent.RETRIEVE
