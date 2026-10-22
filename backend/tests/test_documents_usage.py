from types import SimpleNamespace

from langchain_core.documents import Document

from app.core.usage import UsageTracker, record, tracking
from app.domain.models.chat import TokenUsage
from app.ingestion.splitters import PAGE_BREAK, split_documents
from tests.conftest import new_conversation, stream


def test_usage_is_attributed_to_the_active_stage_only():
    tracker = UsageTracker()
    record("m", TokenUsage(cost_usd=1.0))  # no active tracker: ignored
    with tracking(tracker, "router"):
        record("m", TokenUsage(prompt_tokens=10, cost_usd=0.5))
    with tracking(tracker, "generate"):
        record("n", TokenUsage(completion_tokens=5, cost_usd=0.25))
    assert tracker.breakdown() == {"router": 0.5, "generate": 0.25}
    assert tracker.total.cost_usd == 0.75 and set(tracker.by_model) == {"m", "n"}


def test_splitter_keeps_page_numbers():
    text = PAGE_BREAK.join(["", "Page two text.", "Page three text."])  # first page empty (e.g. a figure)
    chunks = split_documents([Document(page_content=text, metadata={"source": "x.pdf"})])
    assert [(c.page_content, c.metadata["page"]) for c in chunks] == [("Page two text.", 2), ("Page three text.", 3)]


def test_pdf_upload_cites_pages(client, monkeypatch):
    pages = ["Cover page", "", "Dropout zeros a fraction of activations during training."]
    fake = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda t=t: t) for t in pages])
    monkeypatch.setattr("app.api.routes.documents.PdfReader", lambda _bytes: fake)
    res = client.post("/api/v1/documents/", files={"file": ("book.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert res.status_code == 201, res.text
    events = stream(client, new_conversation(client), "What does dropout do during training?")
    citations = next(e["citations"] for e in events if "citations" in e)
    assert citations[0]["page"] == 3


def test_reupload_reuses_unchanged_chunks(client, app):
    body = "First paragraph about dropout.\n\nSecond paragraph about adam."
    doc_id = client.post("/api/v1/documents/", files={"file": ("a.md", body.encode(), "text/markdown")}).json()["id"]
    store = app.state.vector_store
    added = []
    original = store.add_documents
    store.add_documents = lambda docs: (added.extend(docs), original(docs))
    res = client.put(f"/api/v1/documents/{doc_id}", json={"filename": "a.md", "content": body, "tag": "other"})
    assert res.status_code == 200 and added == []


def test_upload_validation(client):
    assert (
        client.post("/api/v1/documents/", files={"file": ("a.exe", b"x", "application/octet-stream")}).status_code
        == 400
    )
    assert client.post("/api/v1/documents/", files={"file": ("a.txt", b"", "text/plain")}).status_code == 400


def test_upload_size_limit(app, client):
    app.state.settings.max_upload_mb = 1
    big = b"a" * (1024 * 1024 + 1)
    assert client.post("/api/v1/documents/", files={"file": ("big.txt", big, "text/plain")}).status_code == 413


def test_delete_document_removes_it_from_retrieval(indexed_client, app):
    doc_id = indexed_client.get("/api/v1/documents/").json()[0]["id"]
    assert indexed_client.delete(f"/api/v1/documents/{doc_id}").status_code == 204
    assert app.state.vector_store.count() == 0 and app.state.bm25_index.size() == 0


def test_cors_is_not_wildcard_with_credentials(client):
    res = client.options(
        "/api/v1/health/", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert res.headers.get("access-control-allow-origin") != "*"
    ok = client.options(
        "/api/v1/health/", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
    )
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
