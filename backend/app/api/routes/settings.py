"""Settings management endpoints."""

from typing import cast
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.schemas.settings import (
    FloatRangeOption,
    GeminiModel,
    IntRangeOption,
    OpenAIModel,
    SettingsOptions,
    SettingsResponse,
    SettingsUpdateRequest,
)
from backend.app.api.dependencies import get_settings
from backend.app.bootstrap import bootstrap_dependencies, refresh_llm_client
from backend.app.core.config import Settings


router = APIRouter()

OPENAI_MODELS: list[OpenAIModel] = ["gpt-4o", "gpt-4o-mini"]
GEMINI_MODELS: list[GeminiModel] = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

OPTIONS = SettingsOptions(
    embedding_providers=["openai", "huggingface"],
    llm_providers=["openai", "gemini"],
    openai_models=OPENAI_MODELS,
    gemini_models=GEMINI_MODELS,
    query_variations=IntRangeOption(min=1, max=10, step=1),
    retrieval_k=IntRangeOption(min=1, max=20, step=1),
    retrieval_lambda=FloatRangeOption(min=0.0, max=1.0, step=0.05),
)

@router.get("", response_model=SettingsResponse)
def read_settings(settings: Settings = Depends(get_settings)) -> SettingsResponse:
    """Return the current runtime settings."""

    return _serialize_settings(settings)


@router.patch("", response_model=SettingsResponse)
def update_settings(
    payload: SettingsUpdateRequest,
    request: Request,
    settings: Settings = Depends(get_settings)
) -> SettingsResponse:
    """Update runtime settings, rebuilding services if required."""

    embedding_changed = False
    llm_changed = False
    reindex_required = False

    if payload.embedding_provider and payload.embedding_provider != settings.embedding_provider:
        if payload.embedding_provider == "openai" and not settings.openai_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot use OpenAI embeddings without configuring OPENAI_API_KEY.",
            )
        settings.embedding_provider = payload.embedding_provider
        embedding_changed = True
        reindex_required = True

    if payload.llm_provider and payload.llm_provider != settings.llm_provider:
        if payload.llm_provider == "gemini" and not settings.gemini_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot use Gemini without configuring GEMINI_API_KEY.",
            )
        settings.llm_provider = payload.llm_provider
        llm_changed = True

        if payload.llm_provider == "gemini":
            if settings.gemini_chat_model not in GEMINI_MODELS:
                settings.gemini_chat_model = GEMINI_MODELS[0]
        else:
            if settings.chat_model not in OPENAI_MODELS:
                settings.chat_model = OPENAI_MODELS[0]

    if payload.chat_model is not None:
        if payload.chat_model != settings.chat_model:
            settings.chat_model = payload.chat_model
            llm_changed = True

    if payload.gemini_chat_model is not None:
        if payload.gemini_chat_model != settings.gemini_chat_model:
            settings.gemini_chat_model = payload.gemini_chat_model
            llm_changed = True

    if payload.query_variations is not None:
        settings.query_variations = payload.query_variations

    if payload.retrieval_k is not None:
        settings.retrieval_k = payload.retrieval_k

    if payload.retrieval_lambda is not None:
        settings.retrieval_lambda = payload.retrieval_lambda

    if embedding_changed or reindex_required:
        bootstrap_dependencies(request.app, settings, force_reindex=reindex_required)
    elif llm_changed:
        refresh_llm_client(request.app, settings)
        request.app.state.settings = settings
    else:
        request.app.state.settings = settings

    return _serialize_settings(settings)

def _serialize_settings(settings: Settings) -> SettingsResponse:
    return SettingsResponse(
        query_variations=settings.query_variations,
        retrieval_k=settings.retrieval_k,
        retrieval_lambda=settings.retrieval_lambda,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        chat_model=cast(OpenAIModel, settings.chat_model),
        gemini_chat_model=cast(GeminiModel, settings.gemini_chat_model),
        options=OPTIONS,
    )