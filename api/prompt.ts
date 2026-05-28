import type { VercelRequest, VercelResponse } from "@vercel/node";
import { runRAG } from "../lib/rag";

export const config = { maxDuration: 60 };

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed." });
  }

  try {
    const question: string = req.body?.question;

    if (!question || typeof question !== "string" || !question.trim()) {
      return res.status(400).json({ error: "Missing or invalid 'question' field in request body." });
    }

    const result = await runRAG(question.trim());
    return res.status(200).json(result);
  } catch (err) {
    console.error("[/api/prompt] Error:", err);
    return res.status(500).json({ error: "Internal server error. Please try again." });
  }
}
