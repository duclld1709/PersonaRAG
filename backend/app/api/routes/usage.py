"""Usage / telemetry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_usage_service
from app.api.schemas.usage import ModelBreakdown, UsageEventResponse, UsageSummaryResponse
from app.services.usage_service import UsageService

router = APIRouter()


@router.get("/summary", response_model=UsageSummaryResponse)
def summary(service: UsageService = Depends(get_usage_service)) -> UsageSummaryResponse:
    data = service.summary()
    return UsageSummaryResponse(
        total_events=data.total_events,
        total_tokens=data.total_tokens,
        total_cost_usd=data.total_cost_usd,
        cache_hits=data.cache_hits,
        avg_latency_ms=data.avg_latency_ms,
        by_model={
            model: ModelBreakdown(**info)  # type: ignore[arg-type]
            for model, info in data.by_model.items()
        },
    )


@router.get("/recent", response_model=list[UsageEventResponse])
def recent(limit: int = 50, service: UsageService = Depends(get_usage_service)) -> list[UsageEventResponse]:
    rows = service.recent(limit=limit)
    return [
        UsageEventResponse(
            model=row["model"],
            mode=row.get("mode"),
            prompt_tokens=row.get("prompt_tokens", 0),
            completion_tokens=row.get("completion_tokens", 0),
            cached_tokens=row.get("cached_tokens", 0),
            cost_usd=row.get("cost_usd", 0.0),
            latency_ms=row.get("latency_ms"),
            cache_hit=bool(row.get("cache_hit", 0)),
            created_at=row["created_at"],
        )
        for row in rows
    ]
