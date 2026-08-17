"""RAG chain: retrieval + context assembly + streaming generation.

The retrieval happens up-front so the UI can display "Source Citations" while
the model is still streaming its answer. Generation uses the GGUF's baked-in
chat template (Qwen / Llama families) via ``llama_cpp`` and yields text
deltas for ``st.write_stream``.
"""

from __future__ import annotations

from typing import Iterator

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are DocuMind, a local study assistant.
Answer the question using ONLY the notes provided below.
If the answer is not in the notes, say exactly:
"I could not find the answer to this question in the provided notes."
Cite your sources: after every statement that comes from a note, mention the
source file and page or line number, e.g. (notes.pdf, p. 3).
Be concise and accurate; never invent facts."""

# Keep the last 8 chat messages so the model stays responsive without
# blowing up the context window.
HISTORY_LIMIT = 8


def format_context(sources: list[tuple[Document, float]]) -> str:
    """Turn retrieved chunks into numbered, citeable context blocks."""
    blocks: list[str] = []
    for i, (doc, score) in enumerate(sources, start=1):
        location = doc.metadata.get("page") or doc.metadata.get("line") or "?"
        label = doc.metadata.get("file_name", "?")
        excerpt = doc.page_content.strip().replace("\n", " ")
        blocks.append(f"[{i}] {label} · p.{location} (score {score:.2f}): {excerpt}")
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Assemble the message list: system prompt, trimmed history, grounded user turn."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in (history or [])[-HISTORY_LIMIT:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    user_content = f"Notes:\n{context}\n\nQuestion: {question}" if context else question
    messages.append({"role": "user", "content": user_content})
    return messages


def generate(
    question: str,
    llm,
    store,
    folder,
    k: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 512,
    history: list[dict] | None = None,
) -> tuple[list[tuple[Document, float]], Iterator[str]]:
    """Retrieve evidence and return ``(sources, stream)``.

    The caller consumes ``stream`` (text deltas) and renders ``sources``
    alongside the answer.
    """
    from src.model_manager import stream_chat
    from src.vector_store import query_vector_store

    sources = query_vector_store(folder, question, k=k)
    context = format_context(sources)
    messages = build_messages(question, context, history)
    return sources, stream_chat(llm, messages, temperature=temperature, max_tokens=max_tokens)