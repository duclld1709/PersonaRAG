"""Application configuration — OpenAI-only, student-research-focused."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Central application settings."""

    app_name: str = "PersonaRAG — Research Assistant for Students"
    environment: str = Field(default="development", alias="APP_ENV")
    api_prefix: str = "/api/v1"

    docs_path: Path = Path("docs")
    vectorstore_path: Path = Path("db") / "chroma_db"
    logs_path: Path = Path("logs")
    database_url: str = Field(default="sqlite:///./db/personarag.db", alias="DATABASE_URL")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    chat_model: str = Field(default="gpt-4.1", alias="OPENAI_CHAT_MODEL")
    utility_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_UTILITY_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL")
    moderation_model: str = Field(default="omni-moderation-latest", alias="OPENAI_MODERATION_MODEL")

    max_conversation_history: int = 20
    query_variations: int = Field(default=3, alias="QUERY_VARIATIONS")
    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K")
    retrieval_fetch_k: int = Field(default=20, alias="RETRIEVAL_FETCH_K")
    hybrid_bm25_weight: float = Field(default=0.4, alias="HYBRID_BM25_WEIGHT")
    hybrid_vector_weight: float = Field(default=0.6, alias="HYBRID_VECTOR_WEIGHT")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    semantic_cache_threshold: float = Field(default=0.95, alias="SEMANTIC_CACHE_THRESHOLD")
    enable_semantic_cache: bool = Field(default=True, alias="ENABLE_SEMANTIC_CACHE")
    rrf_k: int = 60
    vector_search: str = Field(default="mmr", alias="VECTOR_SEARCH")  # mmr | similarity

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    enable_moderation: bool = Field(default=True, alias="ENABLE_MODERATION")
    max_upload_mb: int = Field(default=20, alias="MAX_UPLOAD_MB")
    cors_origins: list[str] = Field(default=["http://localhost:3000"], alias="CORS_ORIGINS")
    max_stream_tokens: int = 1500
    llm_temperature: float = 0.2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()  # type: ignore[call-arg]
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    return settings
