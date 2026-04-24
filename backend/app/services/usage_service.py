"""Usage / telemetry service."""

from __future__ import annotations

from typing import Sequence

from app.infrastructure.persistence.sqlite_usage_repository import (
    SqliteUsageRepository,
    UsageSummary,
)


class UsageService:
    """Read-only facade over the usage repository."""

    def __init__(self, repository: SqliteUsageRepository) -> None:
        self._repository = repository

    def summary(self) -> UsageSummary:
        return self._repository.summary()

    def recent(self, limit: int = 50) -> Sequence[dict]:
        return self._repository.recent(limit=limit)
