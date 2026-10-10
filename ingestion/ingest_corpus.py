"""Bulk-load the shared corpus: every PDF in ingestion/corpus/ -> ChromaDB.

Run from the repo root (venv active):
    python ingestion/ingest_corpus.py                  # all PDFs in ingestion/corpus
    python ingestion/ingest_corpus.py berkshire_1989   # just one (by file stem)
    python ingestion/ingest_corpus.py --no-tables      # skip table detection

Filename convention:  <source>_<year>.pdf   e.g. berkshire_1989.pdf
(book / author / persona are derived from SOURCES in backend/app/services/ingestion.py)
Safe to re-run: each document's old chunks are replaced.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.services.ingestion import IngestError, get_embedder, ingest_pdf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="file stems to ingest (default: all)")
    ap.add_argument("--dir", default=str(REPO / "ingestion" / "corpus"))
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.dir)
    pdfs = sorted(corpus.glob("*.pdf"))
    if args.names:
        wanted = {n.lower().removesuffix(".pdf") for n in args.names}
        pdfs = [p for p in pdfs if p.stem.lower() in wanted]
    if not pdfs:
        sys.exit(f"No PDFs found in {corpus}")

    print("Loading embedding model (first run downloads it)...")
    embedder = get_embedder()

    total_chunks = 0
    for pdf in pdfs:
        t0 = time.time()
        try:
            r = ingest_pdf(
                pdf, scope="global", include_tables=not args.no_tables, embedder=embedder
            )
        except IngestError as e:
            print(f"  FAILED  {pdf.name}: {e}")
            continue
        total_chunks += r.chunks_indexed
        print(
            f"  OK      {pdf.name:<24} {r.pages:>3} pages  {r.chunks_indexed:>4} chunks "
            f"({r.table_chunks} table)  {time.time() - t0:.1f}s"
        )
        for w in r.warnings:
            print(f"          warning: {w}")
    print(f"Done. {total_chunks} chunks indexed.")


if __name__ == "__main__":
    main()
