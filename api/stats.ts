import type { VercelRequest, VercelResponse } from "@vercel/node";
import { RAG_CHUNK_SIZE, RAG_OVERLAP_RATIO, RAG_TOP_K } from "../lib/config";

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.status(200).json({
    chunk_size: RAG_CHUNK_SIZE,
    overlap_ratio: RAG_OVERLAP_RATIO,
    top_k: RAG_TOP_K,
  });
}
