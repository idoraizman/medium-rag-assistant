# Medium Article RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about Medium articles using only the provided dataset — no external knowledge.

## API Endpoints

### POST `/api/prompt`
Query the assistant with a natural language question.

```bash
curl -X POST https://<your-url>/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "List 3 articles about machine learning."}'
```

Response shape:
```json
{
  "response": "...",
  "context": [{ "article_id": "42", "title": "...", "chunk": "...", "score": 0.87 }],
  "Augmented_prompt": { "System": "...", "User": "..." }
}
```

### GET `/api/stats`
Returns current RAG configuration.

```bash
curl https://<your-url>/api/stats
# {"chunk_size":512,"overlap_ratio":0.2,"top_k":15}
```

## Stack
- **Vercel Serverless Functions** (TypeScript)
- **Pinecone** vector database (1536-dim, cosine)
- **Embedding**: `4UHRUIN-text-embedding-3-small` (api.llmod.ai)
- **Generation**: `4UHRUIN-gpt-5-mini` (api.llmod.ai)

## RAG Configuration
| Parameter | Value |
|---|---|
| Chunk size | 512 tokens |
| Overlap ratio | 0.2 |
| Top-k | 15 |

## Setup

### 1. Environment variables
Set the following in Vercel project settings (or `.env.local` for local dev):
```
LLMOD_API_KEY=...
LLMOD_BASE_URL=https://api.llmod.ai/v1
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=medium-articles-agent
```

### 2. Dataset
Download `medium-english-50mb.csv` and place it in the project root.

### 3. Ingest articles (Python)
```bash
cd scripts
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Tune hyperparameters on 300-article subset
.venv/bin/python ingest.py --limit 300 --namespace config-b --chunk-size 512 --overlap 0.2

# Ingest full corpus with final config (run once)
.venv/bin/python ingest.py --namespace main
```

### 4. Deploy
```bash
npx vercel --prod
```
