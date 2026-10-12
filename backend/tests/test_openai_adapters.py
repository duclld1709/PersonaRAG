"""The OpenAI adapters with ChatOpenAI/OpenAIEmbeddings replaced by stubs (no network)."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import BaseModel

from app.core.usage import UsageTracker, tracking
from app.infrastructure.embeddings.openai_embedding import OpenAIEmbeddingClient
from app.infrastructure.llm.openai_chat import OpenAIChatClient

USAGE = {"input_tokens": 1000, "output_tokens": 200, "input_token_details": {"cache_read": 400}}


class Schema(BaseModel):
    queries: list[str]


class StubChatOpenAI:
    def __init__(self, parsing_error=None):
        self.parsing_error = parsing_error

    def invoke(self, messages):
        return AIMessage(content="hello", usage_metadata={**USAGE, "total_tokens": 1200})

    def stream(self, messages):
        yield AIMessageChunk(content="hel")
        yield AIMessageChunk(content="lo")
        yield AIMessageChunk(content="", usage_metadata={**USAGE, "total_tokens": 1200})

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw, "usage is only available on the raw message"
        stub = self

        class _Runnable:
            def invoke(self, messages):
                raw = AIMessage(content="{}", usage_metadata={**USAGE, "total_tokens": 1200})
                parsed = None if stub.parsing_error else schema(queries=["a", "b"])
                return {"raw": raw, "parsed": parsed, "parsing_error": stub.parsing_error}

        return _Runnable()


@pytest.fixture
def client():
    c = OpenAIChatClient("gpt-4.1-mini", api_key="sk-test")
    c._client = StubChatOpenAI()
    return c


# gpt-4.1-mini: $0.40 / 1M input, $0.10 / 1M cached input, $1.60 / 1M output
EXPECTED_COST = (600 * 0.40 + 400 * 0.10 + 200 * 1.60) / 1_000_000


def test_invoke_records_usage_with_cached_tokens(client):
    tracker = UsageTracker()
    with tracking(tracker, "router"):
        response = client.invoke([HumanMessage(content="hi")])
    assert response.content == "hello"
    assert tracker.by_stage["router"].cached_tokens == 400
    assert tracker.total.cost_usd == pytest.approx(EXPECTED_COST)


def test_structured_output_usage_is_recorded(client):
    tracker = UsageTracker()
    with tracking(tracker, "expand"):
        result = client.invoke_structured([HumanMessage(content="q")], Schema)
    assert result.queries == ["a", "b"]
    assert tracker.breakdown() == {"expand": pytest.approx(EXPECTED_COST)}


def test_structured_output_parsing_error_raises(client):
    client._client = StubChatOpenAI(parsing_error=ValueError("bad json"))
    with pytest.raises(ValueError, match="parsing failed"):
        client.invoke_structured([HumanMessage(content="q")], Schema)


def test_stream_yields_deltas_then_usage(client):
    chunks = list(client.stream_with_usage([HumanMessage(content="hi")]))
    assert [d for d, _ in chunks] == ["hel", "lo", ""]
    assert chunks[-1][1].cost_usd == pytest.approx(EXPECTED_COST)
    assert client.snapshot_usage().prompt_tokens == 1000


def test_embedding_usage_is_recorded():
    emb = OpenAIEmbeddingClient("text-embedding-3-large", api_key="sk-test")

    class StubEmbeddings:
        def embed_query(self, text):
            return [0.1, 0.2]

        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    emb._client = StubEmbeddings()
    tracker = UsageTracker()
    with tracking(tracker, "retrieve"):
        emb.embed_query("x" * 400)  # ~100 tokens
    assert tracker.by_model["text-embedding-3-large"].prompt_tokens == 100
