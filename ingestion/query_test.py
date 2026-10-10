"""Checkpoint 2 exit test: raw similarity search, no LLM.

    python ingestion/query_test.py                       # built-in 10 queries
    python ingestion/query_test.py "what is a moat"      # your own query
    python ingestion/query_test.py -k 5 --persona buffett "margin of safety"

Judge by eye: are the passages on-topic, and are book + page correct?
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


def show(query: str, k: int, persona: str | None) -> None:
    print("=" * 100)
    print(f"Q: {query}")
    hits = search(query, k=k, persona_tag=persona)
    if not hits:
        print("  (no results — is the corpus indexed?)")
    for i, h in enumerate(hits, 1):
        tag = " [TABLE]" if h["type"] == "table" else ""
        print(f"\n  {i}. {h['book']}  p.{h['page']}  score={h['score']}{tag}")
        snippet = h["passage"].replace("\n", " ")[:380]
        print(textwrap.indent(textwrap.fill(snippet, 92), "     "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--persona", default=None)
    args = ap.parse_args()
    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    for q in queries:
        show(q, args.k, args.persona)


if __name__ == "__main__":
    main()
