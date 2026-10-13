"""Checkpoint 2 exit test: retrieval only, no answer-writing LLM.

    python ingestion/query_test.py                       # built-in 10 queries
    python ingestion/query_test.py "what is a moat"      # your own query
    python ingestion/query_test.py -k 5 --persona buffett "margin of safety"
    python ingestion/query_test.py --no-rerank           # compare without the reranker

Judge by eye: are the passages on-topic, and are book + page correct?
The bracket shows how each signal ranked a hit: dense (meaning), keyword (exact
words) and rerank (cross-encoder). '-' means that signal did not return it.
"""

import argparse
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.services.retrieval import search  # noqa: E402

DEFAULT_QUERIES = [
    "What is an economic moat?",
    "What does Buffett mean by margin of safety?",
    "How should a company allocate its capital?",
    "Why does Buffett avoid predicting the stock market?",
    "What is intrinsic value and how is it calculated?",
    "What makes a good manager or CEO?",
    "How does insurance float benefit Berkshire?",
    "What does Buffett think about stock buybacks?",
    "Why does Buffett dislike derivatives?",
    "How did Berkshire's performance compare with the S&P 500 in 1974?",  # table query
]


def show(query: str, k: int, persona: str | None, rerank: bool | None) -> None:
    print("=" * 100)
    print(f"Q: {query}")
    hits = search(query, k=k, persona_tag=persona, rerank=rerank)
    if not hits:
        print("  (no results — is the corpus indexed?)")
    for i, h in enumerate(hits, 1):
        tag = " [TABLE]" if h["type"] == "table" else ""
        r = lambda x: f"#{x}" if x else "-"  # noqa: E731
        sig = f"dense {r(h['dense_rank'])} | keyword {r(h['keyword_rank'])} | rerank {r(h['rerank_rank'])}"
        print(f"\n  {i}. {h['book']}  PDF p.{h['page']}  score={h['score']}{tag}   [{sig}]")
        snippet = h["passage"].replace("\n", " ")[:380]
        print(textwrap.indent(textwrap.fill(snippet, 92), "     "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    for q in queries:
        show(q, args.k, args.persona, False if args.no_rerank else None)


if __name__ == "__main__":
    main()
