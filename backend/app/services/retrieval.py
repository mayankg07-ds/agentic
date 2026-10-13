"""Hybrid retrieval over the shared corpus and (optionally) a user's uploads.

Pipeline (no LLM involved):
  1. dense   — embedding similarity from ChromaDB            (meaning)
  2. keyword — BM25 over the same chunks, light stemming     (exact words: "moats", "1974")
  3. fuse    — Reciprocal Rank Fusion of the two lists
  4. rerank  — cross-encoder re-scores the top candidates    (precision)
  5. final order = RRF + reranker rank, so neither signal can be fully ignored

Checkpoint 3's RAG answerer calls `search()` and feeds the passages to the LLM.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Callable, Optional

from app.core.config import get_settings
from app.core.vectorstore import get_collection, get_user_collection
from app.services.ingestion import Embedder, get_embedder

log = logging.getLogger(__name__)

Reranker = Callable[[str, list[str]], list[float]]
RRF_K = 60
RERANK_WEIGHT = 2.0   # the reranker counts as much as dense + keyword combined


# --------------------------------------------------------------------------- keyword (BM25)
_STOP = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "it its this that these those we our us you your he she they them his her their i me my "
    "do does did not no than then so such can could would should will may might has have had "
    "what which who whom why how when where about into over also more most very".split()
)
_SUFFIXES = ("ings", "ing", "edly", "ied", "ies", "ed", "es", "s")


def _stem(w: str) -> str:
    if w.isdigit():
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class _BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        n = max(len(docs), 1)
        self.avgdl = (sum(self.dl) / n) or 1.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str], i: int) -> float:
        tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
        s = 0.0
        for t in set(query):
            f = tf.get(t, 0)
            if f:
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
        return s


_index_cache: dict[str, dict] = {}


def clear_index_cache() -> None:
    """Call after (re)ingesting so the keyword index is rebuilt."""
    _index_cache.clear()


def _index(col) -> dict:
    ids = col.get(include=[])["ids"]
    key = col.name
    cached = _index_cache.get(key)
    if cached and cached["sig"] == (len(ids), hash(tuple(ids))):
        return cached
    data = col.get(include=["documents", "metadatas"])
    idx = {
        "sig": (len(ids), hash(tuple(ids))),
        "ids": data["ids"],
        "docs": data["documents"],
        "metas": data["metadatas"],
        "bm25": _BM25([tokenize(d) for d in data["documents"]]),
    }
    _index_cache[key] = idx
    return idx


def _match(meta: dict, where: Optional[dict]) -> bool:
    return not where or all(meta.get(k) == v for k, v in where.items())


# --------------------------------------------------------------------------- candidates
def _hit(chunk_id: str, doc: str, meta: dict) -> dict:
    return {
        "chunk_id": chunk_id,
        "book": meta["book"],
        "author": meta["author"],
        "page": meta["page"],
        "persona_tag": meta["persona_tag"],
        "type": meta.get("type", "text"),
        "scope": meta.get("scope", "global"),
        "passage": doc,
        "dense_rank": None,
        "keyword_rank": None,
        "rerank_rank": None,
        "rrf": 0.0,
        "score": 0.0,
    }


def _candidates(col, query: str, vec: list[float], n: int, where: Optional[dict]) -> list[dict]:
    """Dense + keyword candidates from one collection, fused with RRF."""
    if col.count() == 0:
        return []
    hits: dict[str, dict] = {}

    kwargs = {"query_embeddings": [vec], "n_results": min(n, col.count())}
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    for rank, (cid, doc, meta) in enumerate(
        zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ):
        h = hits.setdefault(cid, _hit(cid, doc, meta))
        h["dense_rank"] = rank + 1

    idx = _index(col)
    qt = tokenize(query)
    scored = [
        (idx["bm25"].score(qt, i), i)
        for i, m in enumerate(idx["metas"])
        if _match(m, where)
    ]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(reverse=True)
    for rank, (_, i) in enumerate(scored[:n]):
        cid = idx["ids"][i]
        h = hits.setdefault(cid, _hit(cid, idx["docs"][i], idx["metas"][i]))
        h["keyword_rank"] = rank + 1

    for h in hits.values():
        h["rrf"] = sum(1.0 / (RRF_K + r) for r in (h["dense_rank"], h["keyword_rank"]) if r)
    return list(hits.values())


# --------------------------------------------------------------------------- reranker
_reranker: Optional[Reranker] = None


def _to_probs(xs: list[float]) -> list[float]:
    """Cross-encoders return probabilities (sigmoid already applied) or raw logits.
    Decide per batch so a logit that happens to fall in [0,1] is not misread."""
    if xs and min(xs) >= 0.0 and max(xs) <= 1.0:
        return list(xs)
    return [1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0))) for x in xs]


def get_reranker() -> Reranker:
    """Lazy cross-encoder (downloads the model on first use)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(get_settings().rerank_model)
        _reranker = lambda q, docs: [  # noqa: E731
            float(s) for s in model.predict([(q, d) for d in docs], show_progress_bar=False)
        ]
    return _reranker


# --------------------------------------------------------------------------- public API
def search(
    query: str,
    *,
    k: int = 5,
    persona_tag: Optional[str] = None,   # restrict shared corpus to one persona's texts
    owner: Optional[str] = None,         # include this session's uploaded docs
    rerank: Optional[bool] = None,       # default: settings.rerank_enabled
    embedder: Optional[Embedder] = None,
    reranker: Optional[Reranker] = None,
    global_col=None,
    user_col=None,
) -> list[dict]:
    """Top-k passages, best first. `score` is the final relevance in 0..1
    (reranker probability, or normalised fusion score when reranking is off).
    Shared corpus is persona-filtered; uploads are filtered by owner so one
    user's documents never appear for another."""
    s = get_settings()
    n = max(s.retrieve_candidates, k)
    vec = (embedder or get_embedder())([s.embedding_query_prefix + query])[0]

    pool = _candidates(
        global_col or get_collection(), query, vec, n,
        {"persona_tag": persona_tag} if persona_tag else None,
    )
    if owner:
        pool += _candidates(user_col or get_user_collection(), query, vec, n, {"owner": owner})
    pool.sort(key=lambda h: h["rrf"], reverse=True)
    pool = pool[:n]
    if not pool:
        return []

    max_rrf = 2.0 / (RRF_K + 1)
    use_rerank = s.rerank_enabled if rerank is None else rerank
    if use_rerank:
        try:
            scores = (reranker or get_reranker())(query, [h["passage"] for h in pool])
            probs = _to_probs(scores)
            order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
            for r, i in enumerate(order):
                pool[i]["rerank_rank"] = r + 1
                pool[i]["score"] = round(probs[i], 4)
                pool[i]["rrf"] += RERANK_WEIGHT / (RRF_K + r + 1)
        except Exception as e:                       # model missing / offline -> still works
            log.warning("reranker unavailable (%s); using fusion order only", e)
            use_rerank = False
    if not use_rerank:
        for h in pool:
            h["score"] = round(min(h["rrf"] / max_rrf, 1.0), 4)

    pool.sort(key=lambda h: h["rrf"], reverse=True)
    for h in pool:
        h.pop("rrf", None)
    return pool[:k]
