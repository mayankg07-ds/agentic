"""Raw similarity retrieval over the shared corpus and (optionally) a user's uploads.

No LLM involved — this is the Checkpoint 2 exit-criterion layer. Checkpoint 3's
RAG answerer will call `search()` and feed the passages to the LLM.
"""

from __future__ import annotations

from typing import Optional

from app.core.vectorstore import get_collection, get_user_collection
from app.services.ingestion import Embedder, get_embedder


def _query(col, vec: list[float], k: int, where: Optional[dict]) -> list[dict]:
    if col.count() == 0:
        return []
    kwargs = {"query_embeddings": [vec], "n_results": k}
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "chunk_id": meta["chunk_id"],
                "book": meta["book"],
                "author": meta["author"],
                "page": meta["page"],
                "persona_tag": meta["persona_tag"],
                "type": meta.get("type", "text"),
                "scope": meta.get("scope", "global"),
                "passage": doc,
                "score": round(1.0 - float(dist), 4),   # cosine similarity
            }
        )
    return hits


def search(
    query: str,
    *,
    k: int = 5,
    persona_tag: Optional[str] = None,   # restrict shared corpus to one persona's texts
    owner: Optional[str] = None,         # include this session's uploaded docs
    embedder: Optional[Embedder] = None,
    global_col=None,
    user_col=None,
) -> list[dict]:
    """Top-k passages, best first. Shared corpus is persona-filtered; uploads are
    filtered by owner so one user's documents never appear for another."""
    vec = (embedder or get_embedder())([query])[0]
    hits = _query(
        global_col or get_collection(), vec, k,
        {"persona_tag": persona_tag} if persona_tag else None,
    )
    if owner:
        hits += _query(user_col or get_user_collection(), vec, k, {"owner": owner})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]
