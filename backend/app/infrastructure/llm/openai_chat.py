"""OpenAI chat completion adapter."""

from __future__ import annotations

from typing import Iterable, Type, Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.domain.interfaces.llm_client import LLMClient


class OpenAIChatClient(LLMClient):
    """Adapter that wraps LangChain's ChatOpenAI client."""

    def __init__(self, model_name: str, api_key: str | None = None, *, temperature: float = 0.2) -> None:
        self._client = ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage:
        return self._client.invoke(list(messages))

    def stream(self, messages: Iterable[BaseMessage]) -> Iterable[str]:
        for chunk in self._client.stream(list(messages)):
            content = chunk.content
            if content:
                yield content

    def with_structured_output(self, schema: Type[BaseModel]) -> Any:
        return self._client.with_structured_output(schema)
