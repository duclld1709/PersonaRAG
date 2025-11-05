"""Pydantic schemas for configuration settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EmbeddingProvider = Literal["openai", "huggingface"]
LLMProvider = Literal["openai", "gemini"]
OpenAIModel = Literal["gpt-4o", "gpt-4o-mini"]
GeminiModel = Literal["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]


class IntRangeOption(BaseModel):
    """Describe integer slider boundaries."""

    min: int = Field(ge=0)
    max: int = Field(gt=0)
    step: int = Field(default=1, gt=0)


class FloatRangeOption(BaseModel):
    """Describe float slider boundaries."""

    min: float = Field(ge=0.0)
    max: float = Field(gt=0.0)
    step: float = Field(default=0.1, gt=0.0)


class SettingsOptions(BaseModel):
    """Available options that the frontend can present."""

    embedding_providers: list[EmbeddingProvider]
    llm_providers: list[LLMProvider]
    openai_models: list[OpenAIModel]
    gemini_models: list[GeminiModel]
    query_variations: IntRangeOption
    retrieval_k: IntRangeOption
    retrieval_lambda: FloatRangeOption


class SettingsResponse(BaseModel):
    """Current settings with metadata for rendering controls."""

    query_variations: int
    retrieval_k: int
    retrieval_lambda: float
    embedding_provider: EmbeddingProvider
    llm_provider: LLMProvider
    chat_model: OpenAIModel
    gemini_chat_model: GeminiModel
    options: SettingsOptions


class SettingsUpdateRequest(BaseModel):
    """Partial update payload for settings."""

    query_variations: int | None = Field(default=None, ge=1, le=10)
    retrieval_k: int | None = Field(default=None, ge=1, le=20)
    retrieval_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding_provider: EmbeddingProvider | None = None
    llm_provider: LLMProvider | None = None
    chat_model: OpenAIModel | None = None
    gemini_chat_model: GeminiModel | None = None
