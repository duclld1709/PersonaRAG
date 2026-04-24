"""Pydantic schemas for runtime settings.

The new OpenAI-only stack exposes a slimmer surface: just the retrieval and
chat-tuning knobs operators actually adjust. Provider switching is gone.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IntRangeOption(BaseModel):
    min: int = Field(ge=0)
    max: int = Field(gt=0)
    step: int = Field(default=1, gt=0)


class FloatRangeOption(BaseModel):
    min: float = Field(ge=0.0)
    max: float = Field(gt=0.0)
    step: float = Field(default=0.05, gt=0.0)


class SettingsOptions(BaseModel):
    chat_models: list[str]
    utility_models: list[str]
    query_variations: IntRangeOption
    retrieval_k: IntRangeOption
    rerank_top_n: IntRangeOption
    semantic_cache_threshold: FloatRangeOption


class SettingsResponse(BaseModel):
    chat_model: str
    utility_model: str
    embedding_model: str
    query_variations: int
    retrieval_k: int
    rerank_top_n: int
    semantic_cache_threshold: float
    enable_moderation: bool
    options: SettingsOptions


class SettingsUpdateRequest(BaseModel):
    chat_model: str | None = None
    utility_model: str | None = None
    query_variations: int | None = Field(default=None, ge=1, le=10)
    retrieval_k: int | None = Field(default=None, ge=1, le=20)
    rerank_top_n: int | None = Field(default=None, ge=1, le=20)
    semantic_cache_threshold: float | None = Field(default=None, ge=0.5, le=1.0)
    enable_moderation: bool | None = None
