"""OpenAI moderation client — pre-flight safety check on user inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from openai import OpenAI

logger = getLogger(__name__)


@dataclass
class ModerationResult:
    """Outcome of a single moderation call."""

    flagged: bool
    categories: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    @property
    def flagged_categories(self) -> list[str]:
        return [name for name, is_flagged in self.categories.items() if is_flagged]


class OpenAIModerationClient:
    """Thin wrapper around the OpenAI moderation endpoint.

    Fails open (returns not-flagged) on API errors, so a transient outage
    does not block the student's ability to use their own materials. Errors
    are logged for operators.
    """

    def __init__(self, api_key: str | None, model: str = "omni-moderation-latest") -> None:
        self._client = OpenAI(api_key=api_key) if api_key else None
        self._model = model

    def check(self, text: str) -> ModerationResult:
        if not self._client or not text.strip():
            return ModerationResult(flagged=False)
        try:
            response = self._client.moderations.create(model=self._model, input=text)
        except Exception as exc:  # noqa: BLE001 — fail open, log for operator
            logger.warning("Moderation call failed: %s", exc)
            return ModerationResult(flagged=False, error=str(exc))

        result = response.results[0]
        categories: dict[str, bool] = {}
        raw_categories: Any = getattr(result, "categories", None)
        if raw_categories is not None:
            if hasattr(raw_categories, "model_dump"):
                categories = {k: bool(v) for k, v in raw_categories.model_dump().items() if v is not None}
            elif isinstance(raw_categories, dict):
                categories = {k: bool(v) for k, v in raw_categories.items() if v is not None}
        return ModerationResult(flagged=bool(result.flagged), categories=categories)
