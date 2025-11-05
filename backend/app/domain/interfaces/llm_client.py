"""Abstraction for LLM chat completion clients."""

from __future__ import annotations

from typing import Iterable, Protocol, Type, Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class LLMClient(Protocol):
    """Interface for chat LLM interactions."""

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage: ...

    def stream(self, messages: Iterable[BaseMessage]) -> Iterable[str]: ...

    def with_structured_output(self, schema: Type[BaseModel]) -> Any: ...
