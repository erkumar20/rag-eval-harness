from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    groq_api_key: str | None = None
    qdrant_local_path: str = "./qdrant_data"
    database_url: str = "postgresql://localhost:5432/rag_eval"
    baseline_drop_threshold: float = 0.05


@lru_cache
def get_settings() -> Settings:
    return Settings()
