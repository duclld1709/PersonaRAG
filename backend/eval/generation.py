"""End-to-end answer evaluation through the production ChatService.

Nothing is forced: the real router decides retrieve / chitchat / refuse, so a
misrouted in-scope question is counted as the failure it is. (v1's harness hard-wired
every question to retrieval.)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from app.core.prompts import NOT_FOUND_PHRASE, ChatMode
from app.core.usage import UsageTracker, tracking
from app.domain.interfaces.llm_client import LLMClient
from eval.dataset import GoldenQuestion, covers
from eval.judge import judge_answer, scores


@dataclass
class GenerationRecord:
    id: str
    type: str
    question: str
    answer: str
    intent: str
    latency_ms: int
    cost_usd: float
    cost_breakdown: dict[str, float]
    # in-scope questions
    faithfulness: float | None = None
    citation_precision: float | None = None
    correctness: float | None = None
    context_recall: float | None = None
    abstained: bool = False
    # out-of-scope / refuse questions
    behaved_correctly: bool | None = None
    judge: dict = field(default_factory=dict)
    judge_cost_usd: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_generation(app, questions: list[GoldenQuestion], judge_client: LLMClient) -> list[GenerationRecord]:
    chat = app.state.chat_service
    conversations = app.state.conversation_service
    records = []
    for q in questions:
        conversation = conversations.create_conversation(title=f"eval {q.id}")
        start = time.perf_counter()
        result = chat.generate_answer(conversation.id, q.question, mode=ChatMode.ANSWER)
        latency = int((time.perf_counter() - start) * 1000)
        abstained = NOT_FOUND_PHRASE.lower() in result.answer.lower()
        rec = GenerationRecord(
            id=q.id,
            type=q.type,
            question=q.question,
            answer=result.answer,
            intent=result.intent,
            latency_ms=latency,
            cost_usd=result.usage.cost_usd,
            cost_breakdown=result.cost_breakdown,
            abstained=abstained,
        )
        if q.type == "out_of_scope":
            rec.behaved_correctly = abstained or result.intent == "chitchat"
        elif q.type == "refuse":
            rec.behaved_correctly = result.refused
        else:
            rec.context_recall = (
                sum(any(covers(c, e) for c in result.contexts) for e in q.evidence) / len(q.evidence)
                if q.evidence
                else None
            )
            tracker = UsageTracker()
            with tracking(tracker, "judge"):
                verdict = judge_answer(
                    judge_client,
                    question=q.question,
                    reference=q.reference_answer,
                    contexts=result.contexts,
                    answer=result.answer,
                )
            rec.judge = verdict.model_dump()
            rec.judge_cost_usd = tracker.total.cost_usd
            s = scores(verdict)
            rec.faithfulness, rec.citation_precision, rec.correctness = (
                s["faithfulness"],
                s["citation_precision"],
                s["correctness"],
            )
        records.append(rec)
    return records
