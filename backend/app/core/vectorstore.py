"""ChromaDB vector store — local persistent client.

Checkpoint 1: connection + collection handle only.
Checkpoint 2 adds the ingestion pipeline that fills it.
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


def get_collection() -> Collection:
    """Main knowledge-base collection. Chunk metadata contract (Checkpoint 2):
    {book, author, page, chunk_id, persona_tag}."""
    settings = get_settings()
    return get_chroma_client().get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def ping_chroma() -> bool:
    try:
        get_chroma_client().heartbeat()
        return True
    except Exception:
        return False
