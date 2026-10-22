"""Abstraction for LLM chat completion clients."""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, Type, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.domain.models.chat import TokenUsage

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """What the RAG pipeline needs from a chat model (OpenAI in production, fakes in tests)."""

    @property
    def model_name(self) -> str: ...

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage: ...

    def invoke_structured(self, messages: Iterable[BaseMessage], schema: Type[T]) -> T: ...

    def stream_with_usage(self, messages: Iterable[BaseMessage]) -> Iterator[tuple[str, TokenUsage | None]]:
        """Yield ``(delta, None)`` chunks, then ``("", usage)`` once at the end."""
        ...
