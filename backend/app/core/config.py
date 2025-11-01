"""Application configuration powered by pydantic settings."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """Central application settings."""

    app_name: str = "PersonaRAG"
    environment: str = Field(default="development", alias="APP_ENV")
    api_prefix: str = "/api/v1"

    docs_path: Path = Path("docs")
    logs_path: Path = Path("logs")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    
    settings = Settings()
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    settings.logs_path.mkdir(parents=True, exist_ok=True)
    return settings
