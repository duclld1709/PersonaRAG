"""Abstraction for LLM chat completion clients."""

from typing import Any, Iterable, Protocol, Type
from langchain_core.messages import BaseMessage

from pydantic import BaseModel

class LLMClient(Protocol):
    """Interface for chat LLM interactions."""

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage: ...

    def stream(self, messages: Iterable[BaseMessage]) -> Iterable[str]: ...

    def with_structured_output(self, schema: Type[BaseModel]) -> Any: ...


# class LLMClient(Protocol):
#     def invoke(self, messages: Iterable[Any]) -> Any: ...
#     def stream(self, messages: Iterable[Any]) -> Iterable[str]: ...
#     def with_structured_output(self, schema: Type[Any]) -> Any: ...
