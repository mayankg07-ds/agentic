"""PDF ingestion service: PDF -> clean page text + tables -> chunks -> embeddings -> ChromaDB.

One entry point, `ingest_pdf()`, used by BOTH:
  * the bulk loader for the shared corpus   (ingestion/ingest_corpus.py)  scope="global"
  * the user-upload endpoint (Checkpoint 7)                               scope="user"

Design notes
  * Page numbers are PDF page numbers (1-based) so citations are exact.
  * Repeated headers/footers (dates, URLs, "1/37", running titles) are stripped.
  * Tables are detected, converted to Markdown and indexed as their own chunks
    (type="table"), split by rows if big, with the header row repeated.
  * Chunk size comes from settings (default ~1500 chars = ~350 tokens, inside bge-small's
    512-token window). Lower `chunk_max_chars` to ~900 if you switch to MiniLM (256 tokens).
  * Idempotent: re-ingesting the same doc_id replaces its old chunks.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pymupdf  # PyMuPDF

from app.core.config import get_settings
from app.core.vectorstore import get_collection, get_user_collection

Embedder = Callable[[list[str]], list[list[float]]]


# --------------------------------------------------------------------------- errors
class IngestError(Exception):
    """Base class — message is safe to show to the user."""


class NotAPDFError(IngestError):
    pass


class TooLargeError(IngestError):
    pass


class NoTextError(IngestError):
    """PDF has no extractable text (probably scanned images)."""


# --------------------------------------------------------------------------- source registry
# filename convention:  <key>_<year>.pdf   e.g. berkshire_1989.pdf
# Add new sources here as the corpus grows (dalio, ackman, marks, ...).
SOURCES: dict[str, dict[str, str]] = {
    "berkshire": {
        "book": "Berkshire Hathaway Shareholder Letter {year}",
        "author": "Warren Buffett",
        "persona_tag": "buffett",
    },
}


def describe_source(stem: str) -> dict[str, str]:
    """Derive book/author/persona_tag from a filename stem like 'berkshire_1989'."""
    m = re.match(r"^([a-z0-9]+)_(\d{4})$", stem.lower())
    if m and m.group(1) in SOURCES:
        info = SOURCES[m.group(1)]
        return {
            "book": info["book"].format(year=m.group(2)),
            "author": info["author"],
            "persona_tag": info["persona_tag"],
        }
    return {
        "book": stem.replace("_", " ").replace("-", " ").strip().title() or "Untitled",
        "author": "Unknown",
        "persona_tag": "general",
    }


# --------------------------------------------------------------------------- parsing
@dataclass
class TableData:
    caption: str
    rows: list[list[str]]


@dataclass
class PageContent:
    page: int                       # 1-based PDF page number
    text: str                       # prose, tables and noise removed
    tables: list[TableData] = field(default_factory=list)


def _norm_line(line: str) -> str:
    """Normalise a line so 'Page 3/37' and 'Page 4/37' compare equal."""
    return re.sub(r"\s+", " ", re.sub(r"\d", "#", line.lower())).strip()


def _clean_cell(cell: Optional[str]) -> str:
    if cell is None:
        return ""
    if re.fullmatch(r"[.\s·…]*", cell):          # dotted leaders -> empty
        return ""
    cell = re.sub(r"\.{3,}", " ", cell)
    cell = re.sub(r"\s+", " ", cell).strip()
    return cell.replace("|", "/")


_NUMLIKE = re.compile(r"^[\(\-\$]?[\d.,]+%?\)?$")


def _is_real_table(rows: list[list[str]]) -> bool:
    if len(rows) < 3 or max(len(r) for r in rows) < 2:
        return False
    cells = [c for r in rows for c in r if c]
    if len(cells) < 6:
        return False
    return True


_DOTS = re.compile(r"[.\s·…]+")


def _group_lines(words: list) -> list[list]:
    """Group PyMuPDF words into visual lines (left-to-right)."""
    words = sorted(words, key=lambda w: (w[1], w[0]))
    lines: list[list] = []
    for w in words:
        if lines and abs(w[1] - lines[-1][0][1]) < 3:
            lines[-1].append(w)
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w[0])
    return lines


def _is_data_row(line: list) -> bool:
    """A line that ends in >=2 numeric tokens after at most a short label."""
    k = 0
    for w in reversed(line):
        if _NUMLIKE.match(w[4]):
            k += 1
        else:
            break
    return k >= 2 and len(line) - k <= 8


def _row_cells(line: list) -> list[tuple[str, float]]:
    """[(text, x-center)]: leading non-numeric words form one label cell, then one
    cell per numeric token. Works for 'Year . . . 49.5 10.0' style rows."""
    cells: list[tuple[str, float]] = []
    i, label = 0, []
    while i < len(line) and not _NUMLIKE.match(line[i][4]):
        label.append(line[i])
        i += 1
    if label:
        cells.append((" ".join(w[4] for w in label), (label[0][0] + label[-1][2]) / 2))
    cells += [(w[4], (w[0] + w[2]) / 2) for w in line[i:]]
    return cells


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _numeric_tables(page: "pymupdf.Page") -> list[tuple[tuple, list[list[str]], str]]:
    """Unruled numeric tables (e.g. Berkshire's performance table): >=4 consecutive
    data lines. Built directly from words so dotted leaders and odd spacing don't
    create junk columns. Header words are assigned to the nearest value column."""
    words = [w for w in page.get_text("words") if not _DOTS.fullmatch(w[4])]
    lines = _group_lines(words)
    out: list[tuple[tuple, list[list[str]], str]] = []
    i = 0
    while i < len(lines):
        if not _is_data_row(lines[i]):
            i += 1
            continue
        j = i
        while (
            j + 1 < len(lines)
            and _is_data_row(lines[j + 1])
            and lines[j + 1][0][1] - max(w[3] for w in lines[j]) < 14
        ):
            j += 1
        data = lines[i : j + 1]
        i = j + 1
        if len(data) < 4:
            continue
        cells = [_row_cells(r) for r in data]
        widths = Counter(len(c) for c in cells)
        ncols, freq = widths.most_common(1)[0]
        if ncols < 3 or freq / len(cells) < 0.6:
            continue
        # header lines directly above (stop at a gap: that's the title/caption)
        top = lines.index(data[0])
        head: list[list] = []
        k = top
        while k > 0 and len(head) < 6:
            gap = lines[k][0][1] - max(w[3] for w in lines[k - 1])
            if gap >= 7 or _is_data_row(lines[k - 1]):
                break
            head.insert(0, lines[k - 1])
            k -= 1
        centers = [
            _median([c[col][1] for c in cells if len(c) > col]) for col in range(ncols)
        ]
        header_parts: list[list[tuple]] = [[] for _ in range(ncols)]
        spanning: list[str] = []
        for ln in head:
            phrases, cur = [], [ln[0]]
            for w in ln[1:]:
                if w[0] - cur[-1][2] > 12:          # big x-gap -> new phrase
                    phrases.append(cur)
                    cur = [w]
                else:
                    cur.append(w)
            phrases.append(cur)
            for ph in phrases:
                cols = {
                    min(range(ncols), key=lambda c: abs(centers[c] - (w[0] + w[2]) / 2))
                    for w in ph
                }
                if len(cols) >= 2:                  # spans several columns -> table-level note
                    spanning.append(" ".join(w[4] for w in ph))
                else:
                    col = cols.pop()
                    for w in ph:
                        header_parts[col].append((round(w[1] / 3), w[0], w[4]))
        header = [
            " ".join(t for _, _, t in sorted(parts)) or ("" if c == 0 else f"Column {c + 1}")
            for c, parts in enumerate(header_parts)
        ]
        rows = [header] + [[t for t, _ in c] + [""] * (ncols - len(c)) for c in cells]
        rows = [[_clean_cell(x) for x in r[:ncols]] for r in rows]
        used = [w for ln in head + data for w in ln]
        bbox = (
            min(w[0] for w in used) - 4,
            min(w[1] for w in used) - 3,
            max(w[2] for w in used) + 4,
            max(w[3] for w in used) + 3,
        )
        out.append((bbox, rows, " ".join(spanning)))
    return out


def _ruled_tables(page: "pymupdf.Page") -> list[tuple[tuple, list[list[str]], str]]:
    """Tables drawn with ruling lines (PyMuPDF 'lines' strategy)."""
    out = []
    for t in page.find_tables().tables:
        rows = [[_clean_cell(c) for c in r] for r in t.extract()]
        rows = [r for r in rows if any(r)]
        if _is_real_table(rows):
            out.append((tuple(t.bbox), rows, ""))
    return out


def _extract_tables(page: "pymupdf.Page") -> list[tuple[tuple, list[list[str]], str]]:
    """Return [(bbox, rows)] for tables on the page: ruled tables first, then
    unruled numeric tables that don't overlap an already-found table."""
    found: list[tuple[tuple, list[list[str]], str]] = []
    try:
        found = _ruled_tables(page)
    except Exception:
        pass
    try:
        for bbox, rows, note in _numeric_tables(page):
            if not any(_overlaps(bbox, b, 0.3) for b, _, _ in found):
                found.append((bbox, rows, note))
    except Exception:
        pass
    return found


def _overlaps(block_bbox: tuple, table_bbox: tuple, min_frac: float = 0.5) -> bool:
    x0, y0, x1, y1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    ix = max(0.0, min(x1, tx1) - max(x0, tx0))
    iy = max(0.0, min(y1, ty1) - max(y0, ty0))
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    return (ix * iy) / area >= min_frac


def _page_blocks(page: "pymupdf.Page", include_tables: bool):
    """Return (blocks, tables). blocks = [[line, ...], ...] in reading order, table text removed."""
    tables = _extract_tables(page) if include_tables else []
    raw = [b for b in page.get_text("blocks") if b[6] == 0]
    raw.sort(key=lambda b: (round(b[1] / 4), b[0]))
    blocks: list[list[str]] = []
    captions: list[tuple[float, float, str]] = []   # (y0, y1, text) of non-table blocks
    for b in raw:
        bbox = (b[0], b[1], b[2], b[3])
        text = b[4]
        if any(_overlaps(bbox, tb) for tb, _, _ in tables):
            continue
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if lines:
            blocks.append(lines)
            captions.append((b[1], b[3], " ".join(lines)))
    table_objs: list[TableData] = []
    for tb, rows, note in tables:
        cap = ""
        above = [c for c in captions if c[1] <= tb[1] + 2 and tb[1] - c[1] < 90]
        if above:
            cand = max(above, key=lambda c: c[1])[2]
            if len(cand) <= 150:
                cap = re.sub(r"\s+", " ", cand).strip()
        cap = " — ".join(x for x in (cap, note) if x)
        table_objs.append(TableData(caption=cap, rows=rows))
    return blocks, table_objs


def _join_block(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln            # de-hyphenate across line breaks
        else:
            out += " " + ln
    return re.sub(r"\s+", " ", out).strip()


def parse_pdf(
    path: Path, *, include_tables: bool = True, max_pages: Optional[int] = None
) -> list[PageContent]:
    doc = pymupdf.open(path)
    try:
        n = doc.page_count
        if max_pages is not None and n > max_pages:
            raise TooLargeError(f"PDF has {n} pages; the limit is {max_pages}.")
        per_page = [_page_blocks(doc[i], include_tables) for i in range(n)]
    finally:
        doc.close()

    # ---- header/footer detection: first/last 3 lines of a page repeated on many pages
    cand_counter: Counter[str] = Counter()
    for blocks, _ in per_page:
        flat = [ln for b in blocks for ln in b]
        edge = {_norm_line(ln) for ln in flat[:3] + flat[-3:]}
        cand_counter.update(e for e in edge if e)
    threshold = max(3, int(0.4 * n))
    noise = {k for k, v in cand_counter.items() if v >= threshold and len(k) <= 80} if n >= 3 else set()

    pages: list[PageContent] = []
    for i, (blocks, tables) in enumerate(per_page):
        flat_idx = 0
        total = sum(len(b) for b in blocks)
        paragraphs: list[str] = []
        for b in blocks:
            kept = []
            for ln in b:
                is_edge = flat_idx < 3 or flat_idx >= total - 3
                flat_idx += 1
                if is_edge and _norm_line(ln) in noise:
                    continue
                kept.append(ln)
            if kept:
                paragraphs.append(_join_block(kept))
        pages.append(PageContent(page=i + 1, text="\n\n".join(p for p in paragraphs if p), tables=tables))
    return pages


# --------------------------------------------------------------------------- chunking
def _split_long(text: str, max_chars: int) -> list[str]:
    """Split text longer than max_chars: sentences first, then words."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        words, cur = sent.split(" "), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                pieces.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            pieces.append(cur)
    return pieces


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    t = text[-overlap:]
    sp = t.find(" ")
    return t[sp + 1:] if sp != -1 else t


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs into <= max_chars chunks with ~overlap chars carried over."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            units.extend(_split_long(para, max_chars))
    chunks: list[str] = []
    buf = ""
    for u in units:
        if not buf:
            buf = u
        elif len(buf) + 1 + len(u) <= max_chars:
            buf = f"{buf} {u}"
        else:
            chunks.append(buf)
            tail = _tail(buf, overlap)
            buf = f"{tail} {u}" if tail and len(tail) + 1 + len(u) <= max_chars else u
    if buf.strip():
        chunks.append(buf)
    return chunks


def _table_to_markdown(rows: list[list[str]]) -> tuple[str, list[str]]:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = "| " + " | ".join(c or " " for c in rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(c or " " for c in r) + " |" for r in rows[1:]]
    return "\n".join([header, sep]), body


def table_chunks(table: TableData, max_chars: int) -> list[str]:
    """Markdown table pieces <= max_chars, header row repeated, caption prepended."""
    head, body = _table_to_markdown(table.rows)
    prefix = (table.caption + "\n") if table.caption else ""
    out: list[str] = []
    cur: list[str] = []
    base = len(prefix) + len(head) + 1
    size = base
    for row in body:
        if cur and size + len(row) + 1 > max_chars:
            out.append(prefix + head + "\n" + "\n".join(cur))
            cur, size = [], base
        cur.append(row)
        size += len(row) + 1
    if cur:
        out.append(prefix + head + "\n" + "\n".join(cur))
    return out


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    type: str  # "text" | "table"


def build_chunks(
    pages: list[PageContent], doc_id: str, max_chars: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in pages:
        n = 0
        for t in chunk_text(p.text, max_chars, overlap):
            if len(t) < 40:                       # stray fragments / page numbers
                continue
            n += 1
            chunks.append(Chunk(f"{doc_id}-p{p.page:03d}-{n:03d}", t, p.page, "text"))
        for tbl in p.tables:
            for t in table_chunks(tbl, max_chars):
                n += 1
                chunks.append(Chunk(f"{doc_id}-p{p.page:03d}-{n:03d}", t, p.page, "table"))
    return chunks


# --------------------------------------------------------------------------- embeddings
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Lazy local sentence-transformers embedder (downloads the model on first use)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(get_settings().embedding_model)
        _embedder = lambda texts: model.encode(  # noqa: E731
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()
    return _embedder


# --------------------------------------------------------------------------- public API
@dataclass
class IngestResult:
    doc_id: str
    book: str
    scope: str
    pages: int
    chunks_indexed: int
    table_chunks: int
    warnings: list[str] = field(default_factory=list)


def ingest_pdf(
    path: str | Path,
    *,
    scope: str = "global",            # "global" = shared corpus, "user" = an upload
    owner: Optional[str] = None,      # session/user id (required for scope="user")
    doc_id: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    persona_tag: Optional[str] = None,
    include_tables: bool = True,
    embedder: Optional[Embedder] = None,
    collection=None,                   # injectable for tests
) -> IngestResult:
    settings = get_settings()
    path = Path(path)
    if scope not in ("global", "user"):
        raise ValueError("scope must be 'global' or 'user'")
    if scope == "user" and not owner:
        raise ValueError("owner (session id) is required for user uploads")
    if not path.is_file():
        raise IngestError(f"File not found: {path.name}")

    # ---- validation
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"%PDF"):
        raise NotAPDFError("This file is not a valid PDF.")
    if scope == "user" and len(raw) > settings.upload_max_mb * 1024 * 1024:
        raise TooLargeError(f"File is larger than {settings.upload_max_mb} MB.")

    # ---- metadata
    if scope == "global":
        meta = describe_source(path.stem)
        doc_id = doc_id or path.stem.lower()
        owner = ""
    else:
        digest = hashlib.sha1(raw).hexdigest()[:8]
        meta = {
            "book": title or path.stem.replace("_", " ").strip() or "Uploaded document",
            "author": author or "User upload",
            "persona_tag": persona_tag or "user_upload",
        }
        doc_id = doc_id or f"{owner}-{digest}"
    if title and scope == "global":
        meta["book"] = title
    if author and scope == "global":
        meta["author"] = author
    if persona_tag and scope == "global":
        meta["persona_tag"] = persona_tag

    # ---- parse + chunk
    max_pages = settings.upload_max_pages if scope == "user" else None
    pages = parse_pdf(path, include_tables=include_tables, max_pages=max_pages)
    chunks = build_chunks(
        pages, doc_id, settings.chunk_max_chars, settings.chunk_overlap_chars
    )
    warnings: list[str] = []
    empty = sum(1 for p in pages if not p.text and not p.tables)
    if not chunks:
        raise NoTextError(
            "No readable text found. This PDF looks scanned (images only); "
            "OCR is not supported yet."
        )
    if empty:
        warnings.append(f"{empty} of {len(pages)} pages had no extractable text.")

    # ---- embed + index (replace any previous version of this doc)
    col = collection or (get_collection() if scope == "global" else get_user_collection())
    embed = embedder or get_embedder()
    try:
        col.delete(where={"doc_id": doc_id})
    except Exception:
        pass
    B = 64
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        col.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embed([c.text for c in batch]),
            metadatas=[
                {
                    "book": meta["book"],
                    "author": meta["author"],
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "persona_tag": meta["persona_tag"],
                    "type": c.type,
                    "doc_id": doc_id,
                    "scope": scope,
                    "owner": owner or "",
                    "source_file": path.name,
                }
                for c in batch
            ],
        )
    try:                                   # keep the keyword index in sync
        from app.services.retrieval import clear_index_cache

        clear_index_cache()
    except Exception:
        pass
    return IngestResult(
        doc_id=doc_id,
        book=meta["book"],
        scope=scope,
        pages=len(pages),
        chunks_indexed=len(chunks),
        table_chunks=sum(1 for c in chunks if c.type == "table"),
        warnings=warnings,
    )


def delete_document(
    doc_id: str, *, scope: str = "user", owner: Optional[str] = None, collection=None
) -> None:
    """Remove all chunks of a document (used by the upload UI's delete button)."""
    col = collection or (get_user_collection() if scope == "user" else get_collection())
    where: dict = {"doc_id": doc_id}
    if scope == "user":
        if not owner:
            raise ValueError("owner is required to delete a user document")
        where = {"$and": [{"doc_id": doc_id}, {"owner": owner}]}
    col.delete(where=where)