from app.domain.models.chat import Citation
from app.infrastructure.offline import HashEmbeddings
from app.retrieval.cache import SemanticResponseCache
from tests.conftest import new_conversation, stream


def test_scope_isolation_and_live_threshold():
    threshold = {"value": 0.99}
    cache = SemanticResponseCache(HashEmbeddings(), threshold=lambda: threshold["value"])
    cache.store("what is overfitting", "answer A", [Citation(source="s", filename="f")], scope="answer")
    assert cache.lookup("what is overfitting", scope="answer").answer == "answer A"
    assert cache.lookup("what is overfitting", scope="quiz") is None
    assert cache.lookup("what is overfitting exactly in practice", scope="answer") is None
    threshold["value"] = 0.5  # settings panel change applies immediately
    assert cache.lookup("what is overfitting exactly in practice", scope="answer") is not None
    cache.clear()
    assert len(cache) == 0


def _status(events):
    return [e["status"] for e in events if "status" in e]


def test_repeat_first_turn_question_hits_cache(indexed_client):
    stream(indexed_client, new_conversation(indexed_client), "What is overfitting?")
    events = stream(indexed_client, new_conversation(indexed_client), "What is overfitting?")
    assert "cache_hit" in _status(events)
    done = next(e for e in events if "usage" in e)
    assert done["cache_hit"] is True and set(done["cost_breakdown"]) <= {"router", "cache"}
    assert next(e for e in events if "citations" in e)["citations"]


def test_cache_respects_mode(indexed_client):
    stream(indexed_client, new_conversation(indexed_client), "What is overfitting?", mode="answer")
    events = stream(indexed_client, new_conversation(indexed_client), "What is overfitting?", mode="quiz")
    assert "cache_hit" not in _status(events)


def test_follow_up_questions_never_use_cache(indexed_client):
    stream(indexed_client, new_conversation(indexed_client), "What is overfitting?")
    conv = new_conversation(indexed_client)
    stream(indexed_client, conv, "Tell me about cross-validation")
    events = stream(indexed_client, conv, "What is overfitting?")  # same text, but second turn
    assert "cache_hit" not in _status(events)


def test_corpus_change_invalidates_cache(indexed_client, app):
    stream(indexed_client, new_conversation(indexed_client), "What is overfitting?")
    assert len(app.state.response_cache) == 1
    indexed_client.post("/api/v1/documents/", files={"file": ("more.txt", b"Dropout zeros activations.", "text/plain")})
    assert len(app.state.response_cache) == 0


def test_not_found_answers_are_not_cached(indexed_client, app):
    stream(indexed_client, new_conversation(indexed_client), "What is the capital of France?")
    assert len(app.state.response_cache) == 0
