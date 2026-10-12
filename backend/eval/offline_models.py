"""Heuristic stand-ins so the full harness runs in CI without an API key.

Numbers produced with these models are a **smoke test of the plumbing**, not a quality
measurement, and every report generated with ``--offline`` says so.
"""

from __future__ import annotations

import re

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.bootstrap import Providers
from app.core.prompts import (
    NOT_FOUND_PHRASE,
    QUERY_REWRITE_SYSTEM_PROMPT,
    RESEARCH_ASSISTANT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from app.infrastructure.offline import HashEmbeddings, KeywordModeration, ScriptedLLM
from eval.judge import ClaimVerdict, Judgement

_WORD = re.compile(r"\w+")
_STOP = set("the a an of to in is are and or for on with as by that this it be what how why when which does do".split())


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _overlap(a: str, b: str) -> float:
    ta, tb = _terms(a), _terms(b)
    return len(ta & tb) / len(ta) if ta else 0.0


def _system(messages: list[BaseMessage]) -> str:
    return str(messages[0].content) if messages else ""


def _respond(messages: list[BaseMessage]) -> str:
    system, last = _system(messages), str(messages[-1].content)
    if system == ROUTER_SYSTEM_PROMPT:
        return "refuse" if re.search(r"plagiar|detector|turnitin", last, re.I) else "retrieve"
    if system == QUERY_REWRITE_SYSTEM_PROMPT:
        return last.removeprefix("New question: ")
    if system == RESEARCH_ASSISTANT_SYSTEM_PROMPT:
        question = last.split("## Student question", 1)[-1].split("## Excerpts", 1)[0]
        block = last.split("## Excerpts from the student's materials", 1)[-1]
        excerpts = [
            (m.group(1), re.sub(r"^#.*$", "", m.group(2), flags=re.M).strip())
            for m in re.finditer(r"^\[(\d+)\] (.*?)(?=^\[\d+\] |\Z)", block, re.M | re.S)
        ]
        best = max(excerpts, key=lambda e: _overlap(question, e[1]), default=None)
        if not best or _overlap(question, best[1]) < 0.3:
            return f"{NOT_FOUND_PHRASE}. Try uploading a source that covers it."
        sentence = re.split(r"(?<=[.!?])\s", best[1])[0]
        return f"{sentence} [{best[0]}]"
    return "Hello! Upload your course materials and ask me about them."


def _respond_structured(messages: list[BaseMessage], schema: type[BaseModel]) -> BaseModel:
    content = str(messages[-1].content)
    if schema.__name__ == "_QueryVariations":
        return schema(queries=[])
    if schema.__name__ == "_RerankResponse":
        question = content.split("Candidates:", 1)[0]
        items = re.findall(r"^\[(\d+)\] (.+)$", content, re.M)
        return schema(rankings=[{"index": int(i), "score": round(10 * _overlap(question, t), 2)} for i, t in items])
    if schema is Judgement:
        excerpts = content.split("EXCERPTS:", 1)[1].split("ANSWER:", 1)[0]
        answer = content.split("ANSWER:", 1)[1]
        reference = content.split("REFERENCE:", 1)[1].split("EXCERPTS:", 1)[0]
        claims = []
        for sentence in filter(None, (s.strip() for s in re.split(r"(?<=[.!?])\s", answer))):
            cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
            supported = _overlap(sentence, excerpts) >= 0.6
            claims.append(ClaimVerdict(claim=sentence, supported=supported, citations=cited,
                                       citation_correct=supported if cited else None))  # fmt: skip
        overlap = _overlap(reference, answer)
        grade = "correct" if overlap >= 0.5 else "partial" if overlap >= 0.25 else "incorrect"
        return Judgement(claims=claims, correctness=grade, rationale="offline heuristic")
    raise ValueError(f"unexpected schema {schema.__name__}")


def offline_llm(name: str) -> ScriptedLLM:
    return ScriptedLLM(model_name=name, respond=_respond, respond_structured=_respond_structured)


def offline_providers() -> Providers:
    return Providers(
        embedding=HashEmbeddings(),
        chat=offline_llm("offline-chat"),
        utility=offline_llm("offline-utility"),
        moderation=KeywordModeration(),
    )
