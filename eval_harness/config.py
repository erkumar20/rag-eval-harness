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

    # RAGAS metrics need their own LLM/embeddings (separate from the GPT-4o judge in Phase 6).
    # Defaults to the cheaper model for day-to-day dev runs; override to "gpt-4o" for an
    # "official" run before recording results.
    ragas_llm_model: str = "gpt-4o-mini"
    ragas_embedding_model: str = "text-embedding-3-small"


@lru_cache
def get_settings() -> Settings:
    return Settings()
