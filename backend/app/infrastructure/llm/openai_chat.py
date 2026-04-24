"""OpenAI chat adapters with token & cost tracking."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Iterator, Type

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.domain.interfaces.llm_client import LLMClient
from app.domain.models.chat import TokenUsage

from .pricing import estimate_cost


@dataclass
class StreamResult:
    """Captures the full completion plus token accounting after streaming."""

    text: str
    usage: TokenUsage


class OpenAIChatClient(LLMClient):
    """Primary chat adapter returning LangChain-compatible responses.

    Wraps ChatOpenAI so we can:
    - pin temperature / max_tokens
    - accumulate token + cost usage across calls
    - expose streaming with a final usage snapshot
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_usage=True,
        )
        self._cumulative = TokenUsage()
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage:
        response = self._client.invoke(list(messages))
        usage = _usage_from_message(response, self._model_name)
        self._accumulate(usage)
        # Stash usage on the response so the caller can read it.
        if isinstance(response, AIMessage):
            response.response_metadata = {
                **(response.response_metadata or {}),
                "personarag_usage": usage,
            }
        return response

    def stream(self, messages: Iterable[BaseMessage]) -> Iterable[str]:
        for chunk in self._stream_with_usage(messages):
            if chunk:
                yield chunk

    def stream_with_usage(self, messages: Iterable[BaseMessage]) -> Iterator[tuple[str, TokenUsage | None]]:
        """Yield (delta, usage) pairs. usage is non-None on the terminal chunk."""

        usage = TokenUsage()
        for chunk, final_usage in self._stream_internal(messages):
            yield chunk, final_usage
            if final_usage is not None:
                usage = final_usage
        # Nothing more to yield; caller already got usage.

    def _stream_with_usage(self, messages: Iterable[BaseMessage]) -> Iterator[str]:
        for chunk, _ in self._stream_internal(messages):
            yield chunk

    def _stream_internal(self, messages: Iterable[BaseMessage]) -> Iterator[tuple[str, TokenUsage | None]]:
        final_usage: TokenUsage | None = None
        for chunk in self._client.stream(list(messages)):
            content = chunk.content if isinstance(chunk.content, str) else ""
            usage_chunk = getattr(chunk, "usage_metadata", None)
            if usage_chunk:
                final_usage = _usage_from_metadata(usage_chunk, self._model_name)
            yield content, None
        if final_usage:
            self._accumulate(final_usage)
            yield "", final_usage

    def with_structured_output(self, schema: Type[BaseModel]) -> Any:
        return self._client.with_structured_output(schema)

    def snapshot_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._cumulative.prompt_tokens,
                completion_tokens=self._cumulative.completion_tokens,
                cached_tokens=self._cumulative.cached_tokens,
                cost_usd=self._cumulative.cost_usd,
            )

    def _accumulate(self, usage: TokenUsage) -> None:
        with self._lock:
            self._cumulative.add(usage)


class OpenAIUtilityClient(OpenAIChatClient):
    """Cheaper model variant used for routing, rewriting, reranking."""


def _usage_from_message(message: BaseMessage, model: str) -> TokenUsage:
    metadata = getattr(message, "usage_metadata", None) or {}
    return _usage_from_metadata(metadata, model)


def _usage_from_metadata(metadata: dict, model: str) -> TokenUsage:
    prompt_tokens = int(metadata.get("input_tokens", 0) or 0)
    completion_tokens = int(metadata.get("output_tokens", 0) or 0)
    cached_tokens = 0
    details = metadata.get("input_token_details") or {}
    if isinstance(details, dict):
        cached_tokens = int(details.get("cache_read", 0) or 0)
    cost = estimate_cost(
        model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost,
    )
