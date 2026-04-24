"""Usage telemetry schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelBreakdown(BaseModel):
    tokens: int
    cost_usd: float
    events: int


class UsageSummaryResponse(BaseModel):
    total_events: int
    total_tokens: int
    total_cost_usd: float
    cache_hits: int
    avg_latency_ms: float
    by_model: dict[str, ModelBreakdown] = Field(default_factory=dict)


class UsageEventResponse(BaseModel):
    model: str
    mode: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int | None = None
    cache_hit: bool = False
    created_at: str
