import pytest

from eval.dataset import CORPUS_DIR, corpus_files, covers, load_golden, missing_evidence, normalize
from eval.judge import ClaimVerdict, Judgement, scores
from eval.report import bootstrap, percentile
from eval.retrieval import score_ranking


def test_golden_set_is_well_formed():
    questions = load_golden()
    assert len({q.id for q in questions}) == len(questions) >= 80
    assert {q.type for q in questions} == {"single", "multi", "out_of_scope", "refuse"}
    assert all(q.evidence for q in questions if q.in_scope)
    assert not any(q.evidence for q in questions if not q.in_scope)


@pytest.mark.skipif(not (CORPUS_DIR / "d2l").exists(), reason="run `python -m eval.fetch_corpus` first")
def test_every_evidence_quote_occurs_in_the_corpus():
    texts = [p.read_text(encoding="utf-8") for p in corpus_files()]
    assert missing_evidence(load_golden(), texts) == []


def test_normalize_and_partial_cover():
    assert normalize("The *IID*   assumption\n holds") == "the iid assumption holds"
    quote = "the expectation of the gradient remains unchanged and the variance is reduced"
    assert covers("... so the expectation of the gradient remains unchanged and the var", quote)  # 60%+ in chunk
    assert not covers("the expectation of", quote)


def test_score_ranking():
    chunks = ["noise", "contains alpha here", "noise", "contains beta"]
    s = score_ranking(chunks, ("alpha", "beta"), n_relevant_in_index=2, k=3)
    assert s["hit"] == 1.0 and s["recall"] == 0.5 and s["mrr"] == 0.5 and s["ranks"] == [2]
    assert 0 < s["ndcg"] < 1
    assert score_ranking(["x"], ("alpha",), 1, 3)["hit"] == 0.0


def test_judge_scores():
    j = Judgement(
        claims=[
            ClaimVerdict(claim="a", supported=True, citations=[1], citation_correct=True),
            ClaimVerdict(claim="b", supported=False, citations=[2], citation_correct=False),
            ClaimVerdict(claim="c", supported=True),
        ],
        correctness="partial",
    )
    assert scores(j) == {"faithfulness": 2 / 3, "citation_precision": 0.5, "correctness": 0.5, "n_claims": 3}


def test_bootstrap_ci_contains_mean_and_skips_none():
    stat = bootstrap([1, 0, 1, 1, None, 0, 1, 1])
    assert stat.n == 7 and stat.lo <= stat.mean <= stat.hi
    assert bootstrap([]).n == 0 and percentile([1, 2, 3, 4], 50) in (2, 3)
