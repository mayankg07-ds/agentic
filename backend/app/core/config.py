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
    chroma_collection: str = "finance_books"      # shared, persona-tagged corpus
    chroma_user_collection: str = "user_docs"     # user-uploaded PDFs (scoped by owner)

    # --- Embeddings ---
    # NOTE: all-MiniLM-L6-v2 truncates input at 256 word-pieces, so chunks must stay
    # small (see chunk_max_chars). For longer chunks switch to e.g.
    # BAAI/bge-small-en-v1.5 (512 tokens) and raise chunk_max_chars to ~1800.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Ingestion ---
    chunk_max_chars: int = 900        # ~200-230 tokens: safely under MiniLM's 256 limit
    chunk_overlap_chars: int = 150    # ~15% overlap
    upload_max_mb: int = 25
    upload_max_pages: int = 600

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
