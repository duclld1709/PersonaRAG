"""Per-request usage accounting.

Every LLM/embedding call made while a :class:`UsageTracker` is active is added to it,
tagged with the current pipeline stage. Before this existed only the final generation
call was recorded, so the router, query rewrite, query expansion, reranker and
embedding calls were missing from the cost meter and the eval reports.

``tracking()`` must only wrap synchronous code, never a ``yield``: the chat stream is a
generator advanced from a thread pool, and context variables do not survive across
steps there. Streamed generation usage is therefore added explicitly with
:meth:`UsageTracker.add`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.domain.models.chat import TokenUsage

_active: ContextVar[tuple[UsageTracker, str] | None] = ContextVar("usage_tracker", default=None)


@dataclass
class UsageTracker:
    by_stage: dict[str, TokenUsage] = field(default_factory=dict)
    by_model: dict[str, TokenUsage] = field(default_factory=dict)
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage: TokenUsage, stage: str = "other") -> None:
        with self._lock:
            self.calls += 1
            self.by_stage.setdefault(stage, TokenUsage()).add(usage)
            self.by_model.setdefault(model, TokenUsage()).add(usage)

    @property
    def total(self) -> TokenUsage:
        total = TokenUsage()
        for usage in self.by_model.values():
            total.add(usage)
        return total

    def breakdown(self) -> dict[str, float]:
        """USD cost per pipeline stage, for the UI and eval reports."""

        return {name: round(u.cost_usd, 6) for name, u in self.by_stage.items()}


@contextmanager
def tracking(tracker: UsageTracker, stage: str) -> Iterator[None]:
    """Attribute calls made inside the block to ``tracker`` under ``stage``."""

    token = _active.set((tracker, stage))
    try:
        yield
    finally:
        _active.reset(token)


def record(model: str, usage: TokenUsage) -> None:
    """Called by the OpenAI adapters after every non-streaming call."""

    active = _active.get()
    if active is not None:
        tracker, stage = active
        tracker.add(model, usage, stage)
