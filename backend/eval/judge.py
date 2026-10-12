"""LLM-as-judge for answer quality.

One structured call per answer. The judge splits the answer into atomic claims and,
for each claim, decides whether the numbered excerpts support it and whether its
citation markers point at a supporting excerpt. It also grades the answer against the
reference (correct / partial / incorrect).

* faithfulness       = supported claims / claims
* citation precision = cited claims whose citation supports them / cited claims
* correctness        = 1 / 0.5 / 0

Judges have biases, so before trusting these numbers measure agreement with human
labels on a sample (``python -m eval.agreement``). v1 used token-overlap heuristics
and labelled them "RAGAS-compatible".
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.domain.interfaces.llm_client import LLMClient

JUDGE_PROMPT = """You are a strict grader for a retrieval-augmented study assistant.

You get a student QUESTION, a REFERENCE answer written by the course staff, the numbered
EXCERPTS the assistant was given, and the assistant's ANSWER (citations look like [2]).

1. Split the ANSWER into atomic factual claims. Ignore greetings, formatting and the
   "Further study" follow-up questions.
2. For each claim: `supported` = true only if the EXCERPTS (not your own knowledge)
   entail it. `citations` = the numbers cited for it. `citation_correct` = whether at
   least one cited excerpt supports it (null if nothing is cited).
3. `correctness`: compare the ANSWER with the REFERENCE. "correct" if it conveys the
   reference's key points without contradicting it, "partial" if it misses or garbles
   some key point, "incorrect" if it is wrong or does not answer.
Return JSON only."""


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    citations: list[int] = Field(default_factory=list)
    citation_correct: bool | None = None


class Judgement(BaseModel):
    claims: list[ClaimVerdict]
    correctness: Literal["correct", "partial", "incorrect"]
    rationale: str = ""


CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge_answer(client: LLMClient, *, question: str, reference: str, contexts: list[str], answer: str) -> Judgement:
    excerpts = "\n\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(contexts, 1)) or "(none)"
    content = f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nEXCERPTS:\n{excerpts}\n\nANSWER:\n{answer}"
    return client.invoke_structured([SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=content)], Judgement)


def scores(j: Judgement) -> dict[str, float | None]:
    claims = j.claims
    cited = [c for c in claims if c.citations]
    return {
        "faithfulness": sum(c.supported for c in claims) / len(claims) if claims else None,
        "citation_precision": sum(bool(c.citation_correct) for c in cited) / len(cited) if cited else None,
        "correctness": CORRECTNESS_SCORE[j.correctness],
        "n_claims": len(claims),
    }
