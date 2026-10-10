"""ChromaDB vector store — local persistent client.

Two collections, kept separate so user uploads can never leak into the shared corpus:
  * `finance_books` — shared corpus (Berkshire letters, etc.), tagged by persona
  * `user_docs`     — user-uploaded PDFs, scoped by `owner` (session id)

Chunk metadata contract (see API_CONTRACT.md):
  {book, author, page, chunk_id, persona_tag}  +  {type, doc_id, scope, owner, source_file}
"""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import get_settings

_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def _collection(name: str) -> Collection:
    return get_chroma_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection() -> Collection:
    """Shared knowledge-base collection (persona-tagged books and letters)."""
    return _collection(get_settings().chroma_collection)


def get_user_collection() -> Collection:
    """User-uploaded documents collection (filter by `owner`)."""
    return _collection(get_settings().chroma_user_collection)


def ping_chroma() -> bool:
    try:
        get_chroma_client().heartbeat()
        return True
    except Exception:
        return False
