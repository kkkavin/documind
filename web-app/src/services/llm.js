import { getSettings } from "./storage";

export async function generateCompletionStream(prompt, contextText = "", onToken) {
  const settings = getSettings();
  const provider = settings.activeProvider;
  const config = settings.providers?.[provider];

  if (!config) {
    throw new Error(`Provider "${provider}" is not configured in Settings.`);
  }

  if (provider !== "local" && !config.apiKey) {
    throw new Error(`API Key missing for ${provider.toUpperCase()}. Please add it in Settings.`);
  }

  const systemPrompt = `You are StudySync, an AI academic assistant. Answer the student's question based strictly on the provided context below. 

    Instructions:
    - Always state the page number(s) where you found the information.
    - If the answer cannot be found in the provided context, state clearly: "I could not find the answer to this question in the provided notes."
    - Keep your answer clear, accurate, and concise.

    Context:
    ${contextText}`;

  // 1. Google Gemini SSE Stream
  if (provider === "gemini") {
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${config.model}:streamGenerateContent?alt=sse&key=${config.apiKey}`;

    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: `${systemPrompt}\n\nQuestion: ${prompt}` }] }],
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error?.message || `Gemini API Error (HTTP ${response.status})`);
    }

    if (!response.body) throw new Error("No response body received from Gemini API.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep trailing incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
            if (text) onToken(text);
          } catch (e) {
            // Buffer prevents incomplete JSON errors
          }
        }
      }
    }
    return;
  }

  // 2. OpenAI / OpenRouter / Hugging Face / Local Streaming API
  if (["openai", "openrouter", "huggingface", "local"].includes(provider)) {
    let endpoint = "https://api.openai.com/v1/chat/completions";
    let headers = { "Content-Type": "application/json" };

    if (provider === "openrouter") {
      endpoint = "https://openrouter.ai/api/v1/chat/completions";
      headers["Authorization"] = `Bearer ${config.apiKey}`;
      headers["HTTP-Referer"] = window.location.origin;
    } else if (provider === "huggingface") {
      endpoint = "https://api-inference.huggingface.co/v1/chat/completions";
      headers["Authorization"] = `Bearer ${config.apiKey}`;
    } else if (provider === "openai") {
      headers["Authorization"] = `Bearer ${config.apiKey}`;
    } else if (provider === "local") {
      endpoint = `${config.baseUrl.replace(/\/$/, "")}/chat/completions`;
    }

    const response = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: config.model,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: prompt },
        ],
        stream: true,
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error?.message || `${provider.toUpperCase()} API Error (HTTP ${response.status})`);
    }

    if (!response.body) throw new Error("No response body received from API.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ") && trimmed !== "data: [DONE]") {
          try {
            const data = JSON.parse(trimmed.slice(6));
            const content = data.choices?.[0]?.delta?.content;
            if (content) onToken(content);
          } catch (e) {
            // Buffer prevents incomplete JSON errors
          }
        }
      }
    }
    return;
  }

  throw new Error(`Provider "${provider}" is not supported.`);
}