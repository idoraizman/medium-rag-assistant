// RAG hyperparameters — single source of truth.
// GET /api/stats reads from these constants so it always reflects what the code uses.

export const RAG_CHUNK_SIZE = 512;       // tokens per chunk (max 1024)
export const RAG_OVERLAP_RATIO = 0.2;    // 20% overlap (max 0.3)
export const RAG_TOP_K = 15;             // vectors retrieved from Pinecone (max 30)

export const EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small";
export const CHAT_MODEL = "4UHRUIN-gpt-5-mini";
export const PINECONE_NAMESPACE = "main"; // namespace used after full-corpus ingestion
