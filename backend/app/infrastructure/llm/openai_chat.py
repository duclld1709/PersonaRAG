"""OpenAI chat adapters with token & cost tracking."""

from __future__ import annotations

from threading import Lock
from typing import Iterable, Iterator, Type, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core import usage as usage_tracking
from app.domain.interfaces.llm_client import LLMClient
from app.domain.models.chat import TokenUsage

from .pricing import estimate_cost

T = TypeVar("T", bound=BaseModel)


class OpenAIChatClient(LLMClient):
    """Wraps ChatOpenAI to pin generation settings and account for every call.

    Non-streaming calls are reported to the active per-request usage tracker. Streamed
    calls return their usage on the final chunk and the caller adds it (see
    ``app.core.usage``).
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_usage=True,
            reasoning_effort=reasoning_effort,
        )
        self._cumulative = TokenUsage()
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage:
        response = self._client.invoke(list(messages))
        usage = _usage_from_metadata(getattr(response, "usage_metadata", None) or {}, self._model_name)
        self._record(usage)
        if isinstance(response, AIMessage):
            response.response_metadata = {**(response.response_metadata or {}), "personarag_usage": usage}
        return response

    def invoke_structured(self, messages: Iterable[BaseMessage], schema: Type[T]) -> T:
        """Structured output (JSON schema) with usage accounting.

        ``with_structured_output`` alone drops the raw message, and with it the token
        counts, which is why reranker and query-expansion cost used to go unreported.
        """

        result = self._client.with_structured_output(schema, include_raw=True).invoke(list(messages))
        raw = result.get("raw")
        self._record(_usage_from_metadata(getattr(raw, "usage_metadata", None) or {}, self._model_name))
        if result.get("parsing_error") is not None:
            raise ValueError(f"structured output parsing failed: {result['parsing_error']}")
        return result["parsed"]

    def stream_with_usage(self, messages: Iterable[BaseMessage]) -> Iterator[tuple[str, TokenUsage | None]]:
        """Yield ``(delta, None)`` pairs, then ``("", usage)`` once at the end."""

        final_usage: TokenUsage | None = None
        for chunk in self._client.stream(list(messages)):
            content = chunk.content if isinstance(chunk.content, str) else ""
            metadata = getattr(chunk, "usage_metadata", None)
            if metadata:
                final_usage = _usage_from_metadata(metadata, self._model_name)
            if content:
                yield content, None
        if final_usage:
            with self._lock:
                self._cumulative.add(final_usage)
            yield "", final_usage

    def stream(self, messages: Iterable[BaseMessage]) -> Iterable[str]:
        for delta, _ in self.stream_with_usage(messages):
            if delta:
                yield delta

    def snapshot_usage(self) -> TokenUsage:
        with self._lock:
            c = self._cumulative
            return TokenUsage(c.prompt_tokens, c.completion_tokens, c.cached_tokens, c.cost_usd)

    def _record(self, usage: TokenUsage) -> None:
        with self._lock:
            self._cumulative.add(usage)
        usage_tracking.record(self._model_name, usage)


class OpenAIUtilityClient(OpenAIChatClient):
    """Cheaper model variant used for routing, rewriting, reranking."""


def _usage_from_metadata(metadata: dict, model: str) -> TokenUsage:
    prompt_tokens = int(metadata.get("input_tokens", 0) or 0)
    completion_tokens = int(metadata.get("output_tokens", 0) or 0)
    details = metadata.get("input_token_details") or {}
    cached_tokens = int(details.get("cache_read", 0) or 0) if isinstance(details, dict) else 0
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cost_usd=estimate_cost(
            model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cached_tokens=cached_tokens
        ),
    )
