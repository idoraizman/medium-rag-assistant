import openaiClient from "./openai-client";
import { getPineconeIndex } from "./pinecone-client";
import {
  EMBEDDING_MODEL,
  CHAT_MODEL,
  RAG_TOP_K,
  PINECONE_NAMESPACE,
} from "./config";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ContextChunk {
  article_id: string;   // string per assignment spec
  title: string;
  chunk: string;
  score: number;
  authors?: string;
  url?: string;
  timestamp?: string;
  tags?: string;
}

export interface RAGResult {
  response: string;
  context: ContextChunk[];
  Augmented_prompt: {    // capital A per assignment spec
    System: string;
    User: string;
  };
}

// ─── System prompt (verbatim per assignment — do not modify) ──────────────────

const SYSTEM_PROMPT = `You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data." Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Response style: Be concise and direct. For multi-article requests, present results as a numbered list. Always attribute information to a specific article title.`;

// ─── Core RAG functions ───────────────────────────────────────────────────────

/**
 * Embed a query string using the course embedding model.
 */
export async function embedQuery(question: string): Promise<number[]> {
  const response = await openaiClient.embeddings.create({
    model: EMBEDDING_MODEL,
    input: question,
  });
  return response.data[0].embedding;
}

/**
 * Query Pinecone for the top-k most similar chunks.
 */
export async function retrieveChunks(
  embedding: number[],
  topK: number = RAG_TOP_K,
  namespace: string = PINECONE_NAMESPACE
): Promise<ContextChunk[]> {
  const index = getPineconeIndex();
  const ns = index.namespace(namespace);

  const queryResponse = await ns.query({
    vector: embedding,
    topK,
    includeMetadata: true,
  });

  return (queryResponse.matches ?? []).map((match) => ({
    article_id: String(match.metadata?.article_id ?? match.id),
    title: String(match.metadata?.title ?? ""),
    chunk: String(match.metadata?.chunk ?? ""),
    score: match.score ?? 0,
    authors: match.metadata?.authors ? String(match.metadata.authors) : undefined,
    url: match.metadata?.url ? String(match.metadata.url) : undefined,
    timestamp: match.metadata?.timestamp ? String(match.metadata.timestamp) : undefined,
    tags: match.metadata?.tags ? String(match.metadata.tags) : undefined,
  }));
}

/**
 * Deduplicate chunks — keep only the highest-scoring chunk per article_id.
 * This ensures multi-result queries span distinct articles.
 */
export function dedupByArticleId(chunks: ContextChunk[]): ContextChunk[] {
  const seen = new Map<string, ContextChunk>();
  for (const chunk of chunks) {
    const existing = seen.get(chunk.article_id);
    if (!existing || chunk.score > existing.score) {
      seen.set(chunk.article_id, chunk);
    }
  }
  // Return sorted by score descending
  return Array.from(seen.values()).sort((a, b) => b.score - a.score);
}

/**
 * Build the user prompt by injecting retrieved context.
 */
function buildUserPrompt(context: ContextChunk[], question: string): string {
  const contextBlock = context
    .map((c, i) => {
      const meta: string[] = [`[CHUNK ${i + 1}] Article: "${c.title}" (ID: ${c.article_id})`];
      if (c.authors) meta.push(`Authors: ${c.authors}`);
      if (c.timestamp) meta.push(`Published: ${c.timestamp}`);
      if (c.tags) meta.push(`Tags: ${c.tags}`);
      if (c.url) meta.push(`URL: ${c.url}`);
      return `${meta.join(" | ")}\n${c.chunk}`;
    })
    .join("\n\n");

  return `Context from retrieved Medium articles:\n---\n${contextBlock}\n---\n\nQuestion: ${question}`;
}

/**
 * Call the chat model with the augmented prompt and return the full RAG result.
 */
export async function generateAnswer(
  context: ContextChunk[],
  question: string
): Promise<{ response: string; systemPrompt: string; userPrompt: string }> {
  const userPrompt = buildUserPrompt(context, question);

  const completion = await openaiClient.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userPrompt },
    ],
    // gpt-5-mini does not support temperature parameter
  });

  return {
    response: completion.choices[0]?.message?.content ?? "",
    systemPrompt: SYSTEM_PROMPT,
    userPrompt,
  };
}

/**
 * Full RAG pipeline: embed → retrieve → dedup → generate.
 */
export async function runRAG(question: string): Promise<RAGResult> {
  // 1. Embed the question
  const embedding = await embedQuery(question);

  // 2. Retrieve top-k chunks from Pinecone
  const rawChunks = await retrieveChunks(embedding);

  // 3. Deduplicate by article_id (ensures distinct articles for multi-result queries)
  const dedupedChunks = dedupByArticleId(rawChunks);

  // 4. Generate answer using the chat model
  const { response, systemPrompt, userPrompt } = await generateAnswer(
    dedupedChunks,
    question
  );

  return {
    response,
    context: dedupedChunks.map(({ article_id, title, chunk, score }) => ({
      article_id,
      title,
      chunk,
      score,
    })),
    Augmented_prompt: {
      System: systemPrompt,
      User: userPrompt,
    },
  };
}
