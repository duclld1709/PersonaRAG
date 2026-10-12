"""Deterministic stand-ins for the OpenAI clients.

Used by the test suite and by ``python -m eval.run --offline`` (a CI smoke run of the
harness). They exercise the real pipeline code without network access. Their outputs
measure nothing about answer quality, and reports produced with them are labelled as
smoke tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence, Type, TypeVar

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from app.core import usage as usage_tracking
from app.domain.models.chat import TokenUsage

T = TypeVar("T", bound=BaseModel)
_WORD = re.compile(r"\w+", re.UNICODE)


class HashEmbeddings(Embeddings):
    """Bag-of-words feature hashing, L2-normalised. Lexical, not semantic."""

    model_name = "hash-embeddings"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _WORD.findall(text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        usage_tracking.record(self.model_name, TokenUsage(prompt_tokens=sum(len(t) // 4 for t in texts)))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        usage_tracking.record(self.model_name, TokenUsage(prompt_tokens=len(text) // 4))
        return self._vector(text)


Responder = Callable[[list[BaseMessage]], str]
StructuredResponder = Callable[[list[BaseMessage], Type[BaseModel]], BaseModel]


@dataclass
class ScriptedLLM:
    """Chat client whose replies come from plain Python functions.

    Every call reports a fixed token usage, so cost accounting can be tested end to end.
    """

    model_name: str = "scripted"
    respond: Responder = lambda messages: "ok"
    respond_structured: StructuredResponder | None = None
    usage_per_call: TokenUsage = field(
        default_factory=lambda: TokenUsage(prompt_tokens=100, completion_tokens=20, cost_usd=0.001)
    )
    calls: list[str] = field(default_factory=list)

    def invoke(self, messages: Iterable[BaseMessage]) -> BaseMessage:
        messages = list(messages)
        self.calls.append("invoke")
        usage_tracking.record(self.model_name, self._usage())
        return AIMessage(content=self.respond(messages))

    def invoke_structured(self, messages: Iterable[BaseMessage], schema: Type[T]) -> T:
        messages = list(messages)
        self.calls.append(f"structured:{schema.__name__}")
        usage_tracking.record(self.model_name, self._usage())
        if self.respond_structured is None:
            raise ValueError("no structured responder configured")
        return self.respond_structured(messages, schema)  # type: ignore[return-value]

    def stream_with_usage(self, messages: Iterable[BaseMessage]) -> Iterator[tuple[str, TokenUsage | None]]:
        messages = list(messages)
        self.calls.append("stream")
        words = self.respond(messages).split(" ")
        for i, word in enumerate(words):
            yield (word if i == 0 else " " + word), None
        yield "", self._usage()

    def _usage(self) -> TokenUsage:
        u = self.usage_per_call
        return TokenUsage(u.prompt_tokens, u.completion_tokens, u.cached_tokens, u.cost_usd)


class KeywordModeration:
    """Flags only messages containing ``FLAG_ME`` (tests)."""

    @dataclass
    class _Result:
        flagged: bool = False

    def check(self, text: str) -> _Result:
        return self._Result(flagged="FLAG_ME" in text)
