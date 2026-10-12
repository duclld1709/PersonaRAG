import pytest
from langchain_core.documents import Document

from app.infrastructure.offline import ScriptedLLM
from app.retrieval import BM25Index, HybridRetriever, LLMReranker, QueryIntent, QueryRouter, RetrievalHit


def doc(chunk_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id})


class StubVectorStore:
    def __init__(self, ranking: list[Document]):
        self.ranking = ranking
        self.calls = 0

    def as_retriever(self, **kwargs):
        store = self

        class _R:
            def invoke(self, query):
                store.calls += 1
                return store.ranking

        return _R()


DOCS = [doc("a", "dropout zeros activations"), doc("b", "adam optimizer moments"), doc("c", "weight decay penalty")]


def test_bm25_ranks_lexical_matches():
    index = BM25Index()
    index.rebuild(DOCS)
    assert [d.metadata["chunk_id"] for d, _ in index.search("adam moments", top_k=3)] == ["b"]


def test_rrf_fuses_and_weights_sources():
    bm25 = BM25Index()
    bm25.rebuild(DOCS)
    vectors = StubVectorStore([DOCS[2], DOCS[1]])
    hits = HybridRetriever(vector_store=vectors, bm25_index=bm25, vector_weight=0.5, bm25_weight=0.5).retrieve(
        ["adam optimizer"], k=3, fetch_k=10
    )
    assert hits[0].document.metadata["chunk_id"] == "b"  # found by both
    assert hits[0].sources == ("bm25", "vector")


def test_zero_weight_disables_a_branch():
    bm25 = BM25Index()
    bm25.rebuild(DOCS)
    vectors = StubVectorStore(DOCS)
    hits = HybridRetriever(vector_store=vectors, bm25_index=bm25, vector_weight=0, bm25_weight=1).retrieve(
        ["dropout"], k=3, fetch_k=10
    )
    assert vectors.calls == 0 and [h.document.metadata["chunk_id"] for h in hits] == ["a"]
    with pytest.raises(ValueError):
        HybridRetriever(vector_store=vectors, bm25_index=bm25, vector_search="bogus")


def hits_with_scores(*scores):
    return [RetrievalHit(doc(str(i), f"text {i}"), s, ("vector",)) for i, s in enumerate(scores)]


def reranker_returning(rankings):
    class Resp:
        def __init__(self, **kw):
            pass

    def structured(messages, schema):
        return schema(rankings=rankings)

    return LLMReranker(ScriptedLLM(respond_structured=structured))


def test_reranker_rescales_fusion_scores_before_blending():
    # LLM ties hits 0 and 1; the upstream (RRF-scale) score must break the tie.
    hits = hits_with_scores(0.008, 0.016, 0.004)
    reranker = reranker_returning([{"index": 1, "score": 5}, {"index": 2, "score": 5}, {"index": 3, "score": 0}])
    assert [h.document.metadata["chunk_id"] for h in reranker.rerank("q", hits, top_n=2)] == ["1", "0"]


def test_reranker_falls_back_on_error():
    hits = hits_with_scores(0.3, 0.2, 0.1)
    failing = LLMReranker(ScriptedLLM(respond_structured=None))
    assert [h.document.metadata["chunk_id"] for h in failing.rerank("q", hits, top_n=2)] == ["0", "1"]


@pytest.mark.parametrize(
    "reply,expected",
    [("retrieve", QueryIntent.RETRIEVE), ("Chitchat.", QueryIntent.RETRIEVE), ("refuse", QueryIntent.REFUSE),
     ("chitchat", QueryIntent.CHITCHAT), ("", QueryIntent.RETRIEVE)],
)  # fmt: skip
def test_router_parsing(reply, expected):
    assert QueryRouter(ScriptedLLM(respond=lambda m: reply)).classify("hello") == expected


def test_router_defaults_to_retrieve_on_error():
    def boom(messages):
        raise RuntimeError

    assert QueryRouter(ScriptedLLM(respond=boom)).classify("what is x") == QueryIntent.RETRIEVE
