import { Pinecone } from "@pinecone-database/pinecone";

// Singleton Pinecone client — reused across requests in the same serverless instance.
let pineconeClient: Pinecone | null = null;

export function getPineconeClient(): Pinecone {
  if (!pineconeClient) {
    pineconeClient = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });
  }
  return pineconeClient;
}

export function getPineconeIndex() {
  const client = getPineconeClient();
  const indexName = process.env.PINECONE_INDEX_NAME ?? "medium-articles-agent";
  return client.index(indexName);
}
