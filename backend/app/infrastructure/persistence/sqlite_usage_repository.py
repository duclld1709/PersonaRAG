"""SQLite-backed usage/telemetry repository."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from app.domain.models.chat import TokenUsage

from .database import Database


@dataclass
class UsageEvent:
    """One chat pipeline run's usage telemetry."""

    conversation_id: str | None
    mode: str | None
    model: str
    usage: TokenUsage
    latency_ms: int | None = None
    cache_hit: bool = False
    created_at: datetime | None = None
    request_id: str | None = None


@dataclass
class UsageSummary:
    """Aggregate usage across a lookback window."""

    total_events: int
    total_tokens: int
    total_cost_usd: float
    cache_hits: int
    avg_latency_ms: float
    by_model: dict[str, dict[str, float]]


class SqliteUsageRepository:
    """Write & aggregate usage events."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def record(self, event: UsageEvent) -> None:
        created_at = (event.created_at or datetime.utcnow()).isoformat()
        with self._db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO usage_events (
                    conversation_id, mode, model, prompt_tokens, completion_tokens,
                    cached_tokens, cost_usd, latency_ms, cache_hit, created_at, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.conversation_id,
                    event.mode,
                    event.model,
                    event.usage.prompt_tokens,
                    event.usage.completion_tokens,
                    event.usage.cached_tokens,
                    event.usage.cost_usd,
                    event.latency_ms,
                    1 if event.cache_hit else 0,
                    created_at,
                    event.request_id,
                ),
            )

    def summary(self, limit_events: int = 1000) -> UsageSummary:
        with self._db.cursor() as cur:
            cur.execute(
                """
                SELECT id, request_id, model, prompt_tokens, completion_tokens, cached_tokens,
                       cost_usd, latency_ms, cache_hit
                  FROM usage_events
                 ORDER BY id DESC LIMIT ?
                """,
                (limit_events,),
            )
            rows = cur.fetchall()

        total_tokens = 0
        total_cost = 0.0
        cache_hits = 0
        latency_sum = 0
        latency_count = 0
        by_model: dict[str, dict[str, float]] = {}

        for row in rows:
            tokens = row["prompt_tokens"] + row["completion_tokens"]
            total_tokens += tokens
            total_cost += row["cost_usd"] or 0.0
            if row["cache_hit"]:
                cache_hits += 1
            if row["latency_ms"] is not None:
                latency_sum += row["latency_ms"]
                latency_count += 1
            entry = by_model.setdefault(row["model"], {"tokens": 0, "cost_usd": 0.0, "events": 0})
            entry["tokens"] += tokens
            entry["cost_usd"] = round(entry["cost_usd"] + (row["cost_usd"] or 0.0), 6)
            entry["events"] += 1

        avg_latency = latency_sum / latency_count if latency_count else 0.0
        requests = {row["request_id"] or f"row-{row['id']}" for row in rows}
        return UsageSummary(
            total_events=len(requests),
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 6),
            cache_hits=cache_hits,
            avg_latency_ms=round(avg_latency, 1),
            by_model=by_model,
        )

    def recent(self, limit: int = 50) -> Sequence[dict]:
        with self._db.cursor() as cur:
            cur.execute(
                """
                SELECT model, mode, prompt_tokens, completion_tokens, cached_tokens,
                       cost_usd, latency_ms, cache_hit, created_at
                  FROM usage_events
                 ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(row) for row in rows]
