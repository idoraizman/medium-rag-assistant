"""
Medium Articles RAG — Ingestion Script
=======================================
Chunks articles from the CSV, embeds them, and upserts to Pinecone.

Usage examples:
  # Phase A: embed 300-article subset into a named namespace for tuning
  python ingest.py --limit 300 --namespace config-b --chunk-size 512 --overlap 0.2

  # Phase B: embed full corpus into the 'main' namespace (do this ONCE after tuning)
  python ingest.py --namespace main

  # Dry-run — chunk and cache only, no embedding (cost-free)
  python ingest.py --limit 300 --dry-run

Cost gates:
  --dry-run   : chunk + cache only, $0
  --limit N   : embed only the first N articles
  No --limit  : embeds all ~7,600 articles (~$0.27)

The chunks cache (chunks_cache_<namespace>.jsonl) is written after chunking.
If the cache file already exists for the same namespace+chunk_size+overlap,
chunking is skipped and we go straight to embedding/upserting.
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import tiktoken
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR.parent / "medium-english-50mb.csv"

LLMOD_API_KEY = os.environ["LLMOD_API_KEY"]
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-articles-agent")

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100   # vectors per API call
UPSERT_BATCH_SIZE = 100      # vectors per Pinecone upsert

TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[str]:
    """Sliding-window token chunker."""
    tokens = TOKENIZER.encode(text, disallowed_special=())
    stride = max(1, int(chunk_size * (1 - overlap_ratio)))
    chunks = []
    for start in range(0, len(tokens), stride):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(TOKENIZER.decode(chunk_tokens))
        if end >= len(tokens):
            break
    return chunks


def build_cache_path(namespace: str, chunk_size: int, overlap: float) -> Path:
    key = f"{namespace}_{chunk_size}_{overlap}"
    slug = hashlib.md5(key.encode()).hexdigest()[:8]
    return SCRIPT_DIR / f"chunks_cache_{slug}.jsonl"


def load_or_build_chunks(
    df: pd.DataFrame,
    chunk_size: int,
    overlap_ratio: float,
    namespace: str,
) -> list[dict]:
    """
    Load cached chunks if available, otherwise chunk the dataframe and cache.
    Cache is keyed by namespace + chunk_size + overlap_ratio.
    """
    cache_path = build_cache_path(namespace, chunk_size, overlap_ratio)

    if cache_path.exists():
        print(f"[chunk] Cache found at {cache_path.name} — loading...")
        with open(cache_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"[chunk] Loaded {len(records):,} cached chunks.")
        return records

    print(f"[chunk] No cache for namespace='{namespace}' chunk_size={chunk_size} overlap={overlap_ratio}")
    print(f"[chunk] Chunking {len(df):,} articles...")

    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="chunking"):
        article_id = str(idx)
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        chunks = chunk_text(text, chunk_size, overlap_ratio)
        for chunk_idx, chunk in enumerate(chunks):
            records.append({
                "id": f"{article_id}_chunk_{chunk_idx}",
                "article_id": article_id,
                "title": str(row.get("title", "")),
                "authors": str(row.get("authors", "")),
                "url": str(row.get("url", "")),
                "timestamp": str(row.get("timestamp", "")),
                "tags": str(row.get("tags", "")),
                "chunk": chunk,
            })

    # Write cache
    with open(cache_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[chunk] Cached {len(records):,} chunks to {cache_path.name}")
    return records


# ── Embedding & Upserting ─────────────────────────────────────────────────────

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest(
    limit: Optional[int],
    namespace: str,
    chunk_size: int,
    overlap_ratio: float,
    dry_run: bool,
):
    # 1. Load CSV
    print(f"[load] Reading {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH)
    print(f"[load] {len(df):,} articles total.")
    if limit:
        df = df.head(limit)
        print(f"[load] Using first {len(df):,} articles (--limit {limit}).")

    # 2. Chunk (or load from cache)
    records = load_or_build_chunks(df, chunk_size, overlap_ratio, namespace)
    print(f"[chunk] {len(records):,} chunks ready.")

    if dry_run:
        print("[dry-run] Stopping here — no embeddings or upserts (cost $0).")
        return

    # 3. Embed + upsert
    oai = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    total = len(records)
    upserted = 0

    print(f"[embed] Embedding and upserting {total:,} chunks into namespace='{namespace}'...")

    for batch_start in tqdm(range(0, total, EMBEDDING_BATCH_SIZE), desc="batches"):
        batch = records[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        texts = [r["chunk"] for r in batch]

        # Embed
        embeddings = embed_batch(oai, texts)

        # Build Pinecone vectors
        vectors = []
        for record, embedding in zip(batch, embeddings):
            vectors.append({
                "id": record["id"],
                "values": embedding,
                "metadata": {
                    "article_id": record["article_id"],
                    "title": record["title"],
                    "authors": record["authors"],
                    "url": record["url"],
                    "timestamp": record["timestamp"],
                    "tags": record["tags"],
                    "chunk": record["chunk"],
                },
            })

        # Upsert in sub-batches (namespace passed as parameter — Pinecone v9)
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            index.upsert(vectors=vectors[i : i + UPSERT_BATCH_SIZE], namespace=namespace)
            upserted += len(vectors[i : i + UPSERT_BATCH_SIZE])

    print(f"[done] Upserted {upserted:,} vectors into '{PINECONE_INDEX_NAME}' / namespace='{namespace}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Medium articles into Pinecone.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N articles (default: all ~7600).")
    parser.add_argument("--namespace", type=str, default="main",
                        help="Pinecone namespace to upsert into (default: main).")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Token chunk size (default: 512, max: 1024).")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="Overlap ratio (default: 0.2, max: 0.3).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chunk and cache only — no embeddings or upserts.")
    args = parser.parse_args()

    if args.chunk_size > 1024:
        sys.exit("ERROR: --chunk-size cannot exceed 1024 (assignment limit).")
    if args.overlap > 0.3:
        sys.exit("ERROR: --overlap cannot exceed 0.3 (assignment limit).")

    ingest(
        limit=args.limit,
        namespace=args.namespace,
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap,
        dry_run=args.dry_run,
    )
