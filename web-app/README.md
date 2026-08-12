# DocuMind AI

A privacy-first RAG chat app built with React, Vite, and Tailwind CSS. Upload a PDF, index it 100% locally in your browser, and ask questions with streaming answers from your chosen LLM provider (BYOK).

## Getting Started

```bash
npm install
npm run dev
```

## Scripts

- `npm run dev` — start the Vite dev server
- `npm run build` — build for production
- `npm run lint` — run ESLint
- `npm run preview` — preview the production build

## Features

- Zero-server, browser-local PDF indexing (IndexedDB)
- Streaming chat completions with Google Gemini, OpenAI, OpenRouter, Hugging Face, or local Ollama
- BYOK settings saved locally in your browser

## Desktop App

A Streamlit-based companion desktop app lives in `../desktop-app/`. See `requirements.txt` for dependencies and run with `streamlit run app.py`.