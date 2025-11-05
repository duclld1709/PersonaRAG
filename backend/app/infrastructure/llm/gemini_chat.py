"""Google Gemini chat completion adapter."""

from __future__ import annotations

from typing import Any, Iterable, Type

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.domain.interfaces.llm_client import LLMClient


class GeminiChatClient(LLMClient):
    """Adapter that wraps LangChain's ChatGoogleGenerativeAI client."""

    def __init__(self, model_name: str, api_key: str | None = None, *, temperature: float = 0.2) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required to initialize the Gemini client.")
        self._client = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage:
        return self._client.invoke(list(messages))

    def stream(self, messages: Iterable[BaseMessage]) -> Iterable[str]:
        for chunk in self._client.stream(list(messages)):
            content = chunk.content
            if content:
                yield content

    def with_structured_output(self, schema: Type[BaseModel]) -> Any:
        return self._client.with_structured_output(schema)
