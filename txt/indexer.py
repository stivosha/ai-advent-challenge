"""
Text Indexing Pipeline  (GigaChat Embeddings)
==============================================
Strategies:
  1. fixed  -- fixed-size chunks with overlap
  2. struct -- section-based chunks via \\x14...\\x15 heading markers

Embeddings : GigaChat API  (model "Embeddings")
Index      : FAISS (vectors) + SQLite (metadata + text) + JSON (metadata)

Usage:
  python indexer.py                        # build both indexes
  python indexer.py --strategy fixed       # only fixed-size
  python indexer.py --query "red horsemen" # search after indexing
  python indexer.py --query "..." --strategy struct

Credentials:
  Set GIGACHAT_CREDENTIALS env var  OR  create .env file with:
    GIGACHAT_CREDENTIALS=<your_auth_key>
"""

import os
import re
import json
import sqlite3
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOURCE_FILE   = Path(__file__).parent / "horsmen1.txt_Ascii.txt"
ENCODING      = "cp1251"

FIXED_SIZE    = 500   # chars per chunk
FIXED_OVERLAP = 80    # overlap chars between consecutive windows

EMBED_BATCH   = 10    # GigaChat free tier: keep batches small
EMBED_MODEL   = "Embeddings"

INDEX_DIR     = Path(__file__).parent / "index"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str    # e.g. "fixed_0042"
    strategy: str    # "fixed" | "struct"
    source:   str    # filename
    title:    str    # part / book heading
    section:  str    # chapter heading
    text:     str    # chunk text


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------
def load_text(path: Path) -> str:
    return path.read_bytes().decode(ENCODING)


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------
HEADING_RE = re.compile(r'\x14([^\x15]+)\x15')

def parse_headings(text: str) -> List[Tuple[int, str]]:
    """Return [(char_offset, heading_text), ...]"""
    return [(m.start(), m.group(1).strip()) for m in HEADING_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Strategy 1 -- fixed-size chunking
# ---------------------------------------------------------------------------
def chunk_fixed(text: str, source: str) -> List["Chunk"]:
    headings = parse_headings(text)

    def nearest_heading(pos: int) -> Tuple[str, str]:
        title = section = ""
        for h_pos, h_text in headings:
            if h_pos > pos:
                break
            if re.match(r'^\d+[.\s]', h_text) or re.match(r'^\d+$', h_text[:3].strip()):
                section = h_text
            else:
                title = h_text
        return title, section

    clean = HEADING_RE.sub("", text)
    chunks: List[Chunk] = []
    step = FIXED_SIZE - FIXED_OVERLAP
    i = idx = 0
    while i < len(clean):
        fragment = clean[i: i + FIXED_SIZE].strip()
        if fragment:
            title, section = nearest_heading(i)
            chunks.append(Chunk(
                chunk_id=f"fixed_{idx:04d}",
                strategy="fixed",
                source=source,
                title=title,
                section=section,
                text=fragment,
            ))
            idx += 1
        i += step
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 -- structure-based chunking (by headings)
# ---------------------------------------------------------------------------
def chunk_struct(text: str, source: str) -> List["Chunk"]:
    MAX_SECTION = FIXED_SIZE * 2

    headings = parse_headings(text)
    positions = []
    for i, (pos, htxt) in enumerate(headings):
        m = HEADING_RE.search(text, pos)
        content_start = m.end() if m else pos
        content_end   = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        positions.append((htxt, content_start, content_end))

    current_title = ""
    chunks: List[Chunk] = []
    idx = 0

    for htxt, cstart, cend in positions:
        is_chapter = bool(re.match(r'^\d+[.\s]', htxt))
        if is_chapter:
            section = htxt
        else:
            current_title = htxt
            section = ""

        raw = HEADING_RE.sub("", text[cstart:cend]).strip()
        if not raw:
            continue

        # sub-divide oversized sections
        if len(raw) <= MAX_SECTION:
            parts = [raw]
        else:
            parts = [raw[j: j + MAX_SECTION].strip()
                     for j in range(0, len(raw), MAX_SECTION)]

        for part_i, sub in enumerate(parts):
            if not sub:
                continue
            label = (f"{section} [{part_i+1}/{len(parts)}]"
                     if len(parts) > 1 else section)
            chunks.append(Chunk(
                chunk_id=f"struct_{idx:04d}",
                strategy="struct",
                source=source,
                title=current_title,
                section=label,
                text=sub,
            ))
            idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Embeddings via GigaChat
# ---------------------------------------------------------------------------
def embed(chunks: List[Chunk]) -> np.ndarray:
    creds = os.environ.get("GIGACHAT_API_KEY")
    if not creds:
        raise RuntimeError(
            "GIGACHAT_API_KEY not set. "
            "Add it to .env or export as environment variable."
        )

    texts = [c.text for c in chunks]
    all_vecs: List[List[float]] = []

    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start: start + EMBED_BATCH]
            resp  = giga.embeddings(batch, model=EMBED_MODEL)
            batch_vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
            all_vecs.extend(batch_vecs)
            done = min(start + EMBED_BATCH, len(texts))
            print(f"  embedded {done}/{len(texts)}", end="\r", flush=True)

    print()
    return np.array(all_vecs, dtype="float32")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_index(strategy: str, chunks: List[Chunk], vecs: np.ndarray) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    dim = vecs.shape[1]

    # FAISS
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    faiss.write_index(index, str(INDEX_DIR / f"{strategy}.faiss"))

    # SQLite
    db_path = INDEX_DIR / f"{strategy}.db"
    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute("""
        CREATE TABLE chunks (
            rowid    INTEGER PRIMARY KEY,
            chunk_id TEXT, strategy TEXT,
            source   TEXT, title    TEXT,
            section  TEXT, text     TEXT
        )
    """)
    con.executemany(
        "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
        [(i, c.chunk_id, c.strategy, c.source, c.title, c.section, c.text)
         for i, c in enumerate(chunks)]
    )
    con.commit()
    con.close()

    # JSON metadata (no vectors)
    (INDEX_DIR / f"{strategy}_meta.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"  Saved: index/{strategy}.faiss | .db | _meta.json")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(query: str, strategy: str, top_k: int = 5) -> None:
    faiss_path = INDEX_DIR / f"{strategy}.faiss"
    db_path    = INDEX_DIR / f"{strategy}.db"
    if not faiss_path.exists():
        print(f"Index not found: {faiss_path}. Run without --query first.")
        return

    creds = os.environ.get("GIGACHAT_API_KEY")
    with GigaChat(credentials=creds, verify_ssl_certs=False) as giga:
        resp  = giga.embeddings([query], model=EMBED_MODEL)
        q_vec = np.array([resp.data[0].embedding], dtype="float32")

    index = faiss.read_index(str(faiss_path))
    distances, ids = index.search(q_vec, top_k)

    con = sqlite3.connect(db_path)
    print(f"\n=== Top-{top_k} [{strategy}] for: '{query}' ===")
    for rank, (dist, row_id) in enumerate(zip(distances[0], ids[0]), 1):
        row = con.execute(
            "SELECT chunk_id, title, section, text FROM chunks WHERE rowid=?",
            (int(row_id),)
        ).fetchone()
        if row:
            cid, title, section, text = row
            print(f"\n[{rank}] dist={dist:.4f}  id={cid}")
            print(f"     title:   {title}")
            print(f"     section: {section}")
            print(f"     text:    {text[:200].strip()}...")
    con.close()


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------
def compare(fixed_chunks: List[Chunk], struct_chunks: List[Chunk]) -> None:
    def stats(chunks):
        lens = np.array([len(c.text) for c in chunks])
        return {
            "count": int(len(lens)),
            "min":   int(lens.min()),
            "max":   int(lens.max()),
            "mean":  round(float(lens.mean()), 1),
            "std":   round(float(lens.std()), 1),
        }

    report = {
        "fixed":  stats(fixed_chunks),
        "struct": stats(struct_chunks),
        "notes": {
            "fixed": (
                f"Uniform windows of {FIXED_SIZE} chars, overlap {FIXED_OVERLAP}. "
                "Predictable size, may cut mid-sentence. "
                "Good for dense retrieval."
            ),
            "struct": (
                "One chunk per chapter/section (sub-divided if > 2x fixed_size). "
                "Preserves semantic units. "
                "Better provenance and attribution."
            ),
        },
    }

    report_path = INDEX_DIR / "comparison.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n-- Chunking comparison --")
    for strat in ("fixed", "struct"):
        s = report[strat]
        print(
            f"  {strat:6s}: {s['count']} chunks  "
            f"len=[{s['min']}..{s['max']}]  "
            f"mean={s['mean']} +/- {s['std']}"
        )
    print(f"  Report saved -> index/comparison.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Text indexing pipeline (GigaChat)")
    parser.add_argument("--strategy", default="both",
                        choices=["fixed", "struct", "both"],
                        help="Chunking strategy to build (default: both)")
    parser.add_argument("--query", default=None,
                        help="Search query to run after indexing")
    args = parser.parse_args()

    print(f"Loading: {SOURCE_FILE.name}")
    text   = load_text(SOURCE_FILE)
    source = SOURCE_FILE.name
    print(f"  {len(text):,} chars  |  encoding: {ENCODING}")

    strategies = ["fixed", "struct"] if args.strategy == "both" else [args.strategy]
    all_chunks: dict = {}

    for strat in strategies:
        print(f"\n-- Strategy: {strat} --")
        t0 = time.time()

        chunks = chunk_fixed(text, source) if strat == "fixed" else chunk_struct(text, source)
        all_chunks[strat] = chunks
        print(f"  {len(chunks)} chunks  ({time.time()-t0:.1f}s)")

        print("  Generating embeddings via GigaChat...")
        t1 = time.time()
        vecs = embed(chunks)
        print(f"  shape={vecs.shape}  ({time.time()-t1:.1f}s)")

        print("  Saving index...")
        save_index(strat, chunks, vecs)

    # Comparison report (build missing strategy in-memory, no embeddings needed)
    if "fixed" not in all_chunks:
        all_chunks["fixed"] = chunk_fixed(text, source)
    if "struct" not in all_chunks:
        all_chunks["struct"] = chunk_struct(text, source)
    compare(all_chunks["fixed"], all_chunks["struct"])

    if args.query:
        for strat in strategies:
            search(args.query, strat)


if __name__ == "__main__":
    main()
