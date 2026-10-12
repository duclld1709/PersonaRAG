"""Golden set and corpus loading, plus evidence-based relevance labels.

Each golden question lists ``evidence``: verbatim quotes from the corpus that answer it.
A retrieved chunk is *relevant* to a question when it contains one of those quotes
(after normalising whitespace and Markdown emphasis). Because the quotes are matched,
not the file names, a lecture-note chunk and a textbook chunk that state the same fact
both count. v1 scored retrieval by file name over a two-file corpus, where any chunk
counted as a hit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

EVAL_DIR = Path(__file__).parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_FILE = EVAL_DIR / "golden_qa.jsonl"

# A quote split across two chunks still counts if this share of it sits in one chunk.
PARTIAL_MATCH = 0.6


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    type: str  # single | multi | out_of_scope | refuse
    question: str
    reference_answer: str
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def in_scope(self) -> bool:
        return self.type in ("single", "multi")


def normalize(text: str) -> str:
    text = re.sub(r"(?<!\*)\*{1,2}(?!\*)", "", text)  # *emphasis* / **bold**
    return " ".join(text.split()).lower()


def load_golden(path: Path = GOLDEN_FILE) -> list[GoldenQuestion]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(GoldenQuestion(d["id"], d["type"], d["question"], d["reference_answer"], tuple(d["evidence"])))
    return out


def corpus_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    files = sorted(corpus_dir.glob("*.md")) + sorted((corpus_dir / "d2l").glob("*.md"))
    if not (corpus_dir / "d2l").exists():
        raise FileNotFoundError("textbook corpus missing: run `python -m eval.fetch_corpus` first")
    return files


def fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def covers(chunk: str, quote: str) -> bool:
    """Whether ``chunk`` contains ``quote`` (fully, or >= PARTIAL_MATCH of it contiguously)."""

    c, q = normalize(chunk), normalize(quote)
    if q in c:
        return True
    # Any contiguous match covering >= 60% of the quote contains its middle 20%.
    if q[int(0.4 * len(q)) : int(0.6 * len(q))] not in c:
        return False
    match = SequenceMatcher(None, c, q, autojunk=False).find_longest_match(0, len(c), 0, len(q))
    return match.size >= PARTIAL_MATCH * len(q)


def missing_evidence(questions: list[GoldenQuestion], texts: list[str]) -> list[tuple[str, str]]:
    """Quotes that do not occur anywhere in the corpus (annotation errors)."""

    corpus = normalize("\n".join(texts))
    return [(q.id, e) for q in questions for e in q.evidence if normalize(e) not in corpus]
