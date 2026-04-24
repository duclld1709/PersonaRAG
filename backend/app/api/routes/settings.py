"""Settings management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_settings
from app.api.schemas.settings import (
    FloatRangeOption,
    IntRangeOption,
    SettingsOptions,
    SettingsResponse,
    SettingsUpdateRequest,
)
from app.bootstrap import refresh_llm_client
from app.core.config import Settings

router = APIRouter()

CHAT_MODELS: list[str] = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]
UTILITY_MODELS: list[str] = ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]

OPTIONS = SettingsOptions(
    chat_models=CHAT_MODELS,
    utility_models=UTILITY_MODELS,
    query_variations=IntRangeOption(min=1, max=8, step=1),
    retrieval_k=IntRangeOption(min=1, max=20, step=1),
    rerank_top_n=IntRangeOption(min=1, max=10, step=1),
    semantic_cache_threshold=FloatRangeOption(min=0.80, max=1.0, step=0.01),
)


@router.get("", response_model=SettingsResponse)
def read_settings(settings: Settings = Depends(get_settings)) -> SettingsResponse:
    return _serialize_settings(settings)


@router.patch("", response_model=SettingsResponse)
def update_settings(
    payload: SettingsUpdateRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> SettingsResponse:
    llm_changed = False

    if payload.chat_model and payload.chat_model in CHAT_MODELS and payload.chat_model != settings.chat_model:
        settings.chat_model = payload.chat_model
        llm_changed = True
    if (
        payload.utility_model
        and payload.utility_model in UTILITY_MODELS
        and payload.utility_model != settings.utility_model
    ):
        settings.utility_model = payload.utility_model
        llm_changed = True
    if payload.query_variations is not None:
        settings.query_variations = payload.query_variations
    if payload.retrieval_k is not None:
        settings.retrieval_k = payload.retrieval_k
    if payload.rerank_top_n is not None:
        settings.rerank_top_n = payload.rerank_top_n
    if payload.semantic_cache_threshold is not None:
        settings.semantic_cache_threshold = payload.semantic_cache_threshold
    if payload.enable_moderation is not None:
        settings.enable_moderation = payload.enable_moderation

    if llm_changed:
        refresh_llm_client(request.app, settings)

    request.app.state.settings = settings
    return _serialize_settings(settings)


def _serialize_settings(settings: Settings) -> SettingsResponse:
    return SettingsResponse(
        chat_model=settings.chat_model,
        utility_model=settings.utility_model,
        embedding_model=settings.embedding_model,
        query_variations=settings.query_variations,
        retrieval_k=settings.retrieval_k,
        rerank_top_n=settings.rerank_top_n,
        semantic_cache_threshold=settings.semantic_cache_threshold,
        enable_moderation=settings.enable_moderation,
        options=OPTIONS,
    )
