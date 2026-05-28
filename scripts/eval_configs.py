"""
Evaluate 3 Pinecone namespace configs against the 4 assignment query types.
Prints top-3 retrieved chunks per query per config — no LLM call, $0 cost.

Usage:
  python eval_configs.py
"""

import os
from openai import OpenAI
from pinecone import Pinecone

LLMOD_API_KEY  = os.environ["LLMOD_API_KEY"]
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-articles-agent")
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"

QUERIES = [
    ("Q1 – Precise fact",   "An article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable"),
    ("Q2 – Multi-topic",    "Articles about education and learning"),
    ("Q3 – Key idea",       "Article arguing past pandemics like the bubonic plague can spur innovation and economic recovery"),
    ("Q4 – Recommendation", "Practical beginner-friendly advice on building habits that actually stick"),
]

NAMESPACES = ["config-a", "config-b", "config-c"]
TOP_K = 5


def embed(client: OpenAI, text: str) -> list[float]:
    r = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


def main():
    oai = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
    pc  = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX_NAME)

    for q_label, q_text in QUERIES:
        print(f"\n{'='*70}")
        print(f"  {q_label}")
        print(f"  Query: {q_text}")
        print(f"{'='*70}")

        vec = embed(oai, q_text)

        for ns in NAMESPACES:
            res = idx.query(vector=vec, top_k=TOP_K, include_metadata=True, namespace=ns)
            matches = res.matches or []

            # Deduplicate by article_id, keep highest score
            seen: dict[str, dict] = {}
            for m in matches:
                aid = str(m.metadata.get("article_id", m.id))
                if aid not in seen or m.score > seen[aid]["score"]:
                    seen[aid] = {"score": m.score, "title": m.metadata.get("title",""), "chunk": m.metadata.get("chunk","")[:120]}

            top = sorted(seen.values(), key=lambda x: -x["score"])
            print(f"\n  [{ns}]")
            for i, r in enumerate(top, 1):
                print(f"    {i}. score={r['score']:.4f}  title={r['title']!r}")
                print(f"       chunk: {r['chunk']!r}...")


if __name__ == "__main__":
    main()
