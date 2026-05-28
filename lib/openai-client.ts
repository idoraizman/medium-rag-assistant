import OpenAI from "openai";

// OpenAI-compatible client pointing at the course-provided API endpoint.
const openaiClient = new OpenAI({
  apiKey: process.env.LLMOD_API_KEY!,
  baseURL: process.env.LLMOD_BASE_URL ?? "https://api.llmod.ai/v1",
});

export default openaiClient;
