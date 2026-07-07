"""Central configuration — all secrets/settings come from environment variables (.env)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = backend/app/core/config.py -> 3 levels up
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Oracle Finance API"
    debug: bool = True
    cors_origins: str = "http://localhost:5173"

    # --- LLM (NVIDIA NIM, OpenAI-compatible) ---
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b"

    # --- MongoDB ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "oracle_finance"

    # --- ChromaDB ---
    chroma_persist_dir: str = str(BASE_DIR / "chroma_data")
    chroma_collection: str = "finance_books"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
