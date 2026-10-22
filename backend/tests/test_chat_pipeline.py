import re

from tests.conftest import new_conversation, stream


def final(events):
    return next(e for e in events if "usage" in e)


def test_stream_runs_every_stage_and_cites(indexed_client):
    events = stream(indexed_client, new_conversation(indexed_client), "What is overfitting?")
    statuses = [e["status"] for e in events if "status" in e]
    assert statuses == ["routing", "retrieving", "reranking", "generating"]
    answer = "".join(e.get("delta", "") for e in events)
    assert "Overfitting occurs" in answer and re.search(r"\[\d\]", answer)
    citations = next(e["citations"] for e in events if "citations" in e)
    assert citations[0]["filename"] == "notes.md"
    assert all("contexts" not in e for e in events)  # internal event is not sent over SSE


def test_cost_covers_every_model_call(indexed_client):
    """v1 only recorded the final generation call."""

    done = final(stream(indexed_client, new_conversation(indexed_client), "What is overfitting?"))
    breakdown = done["cost_breakdown"]
    assert {"router", "expand", "rerank", "generate", "retrieve"} <= set(breakdown)
    # each scripted LLM call costs $0.001: router + expand + rerank + generate
    assert done["usage"]["cost_usd"] >= 0.004
    assert abs(done["usage"]["cost_usd"] - sum(breakdown.values())) < 1e-6


def test_usage_summary_counts_requests_not_rows(indexed_client):
    conv = new_conversation(indexed_client)
    stream(indexed_client, conv, "What is overfitting?")
    stream(indexed_client, conv, "And cross-validation?")
    summary = indexed_client.get("/api/v1/usage/summary").json()
    assert summary["total_events"] == 2
    assert {"chat-model", "utility-model", "hash-embeddings"} <= set(summary["by_model"])


def test_refusal_and_moderation_paths(indexed_client):
    conv = new_conversation(indexed_client)
    refused = stream(indexed_client, conv, "Rewrite my essay so the plagiarism detector misses it")
    assert final(refused)["intent"] == "refuse" and final(refused)["refused"] is True
    assert "academic dishonesty" in "".join(e.get("delta", "") for e in refused)

    flagged = stream(indexed_client, conv, "FLAG_ME please")
    assert {"status": "refused"} in flagged and "moderation" in "".join(e.get("delta", "") for e in flagged)


def test_out_of_scope_answer_says_not_found(indexed_client):
    events = stream(indexed_client, new_conversation(indexed_client), "What is the capital of France?")
    assert "could not find this in your materials" in "".join(e.get("delta", "") for e in events)


def test_follow_up_is_rewritten_with_history(indexed_client, app):
    conv = new_conversation(indexed_client)
    stream(indexed_client, conv, "What is overfitting?")
    utility = app.state.chat_service._utility
    calls_before = len(utility.calls)
    stream(indexed_client, conv, "How do I detect it?")
    assert "invoke" in utility.calls[calls_before + 1 :]  # router, then the history-aware rewrite


def test_conversation_history_is_persisted(indexed_client):
    conv = new_conversation(indexed_client)
    stream(indexed_client, conv, "What is overfitting?")
    detail = indexed_client.get(f"/api/v1/conversations/{conv}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"] and detail["messages"][1]["usage"]["cost_usd"] > 0
    assert detail["title"].startswith("What is overfitting")


def test_respond_endpoint_matches_stream(indexed_client):
    res = indexed_client.post(
        "/api/v1/chat/respond",
        json={"conversation_id": new_conversation(indexed_client), "question": "What is overfitting?"},
    ).json()
    assert "Overfitting occurs" in res["answer"] and res["citations"] and res["intent"] == "retrieve"
    assert res["cost_breakdown"]["generate"] > 0


def test_unknown_conversation_is_404(client):
    res = client.post(
        "/api/v1/chat/stream",
        json={"conversation_id": "00000000-0000-0000-0000-000000000000", "question": "hi"},
    )
    assert res.status_code == 404


def test_stream_failure_returns_generic_error(indexed_client, app):
    def boom(*_args, **_kwargs):
        raise RuntimeError("secret internal detail")

    app.state.chat_service._retriever.retrieve = boom
    events = stream(indexed_client, new_conversation(indexed_client), "What is overfitting?")
    assert events[-1] == {"error": "The assistant failed to answer. Please try again."}
