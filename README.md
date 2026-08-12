# DocuMind AI

A privacy-first RAG (Retrieval-Augmented Generation) document assistant. Upload PDF notes or textbooks, index them, and ask questions with AI answers that cite the page numbers where the information was found.

## Architecture

The repository is a monorepo with two application variants sharing the same RAG concept:

| App | Stack | Description |
| --- | --- | --- |
| `web-app/` | React, Vite, Tailwind CSS | Zero-server browser app. PDFs are parsed, chunked, and indexed 100% locally in the browser (IndexedDB via Dexie). Streaming chat completions via BYOK (bring-your-own-key) providers. |
| `desktop-app/` | Python, Streamlit, LangChain | Companion desktop tool. Uses LangChain with a Chroma vector store and sentence-transformer embeddings, with Gemini generating the answers. |

## Features

- **Privacy-first**: Web app runs entirely client-side — no server, no upload of your documents. Indexed data never leaves the browser.
- **PDF indexing**: Upload a PDF and it is parsed, chunked, and stored locally.
- **Source-cited answers**: The assistant always states the page number(s) where it found the information.
- **Multiple LLM providers (web)**: Google Gemini, OpenAI, OpenRouter, Hugging Face, or a local Ollama endpoint.
- **BYOK settings**: API keys and model choices are saved only in your browser's localStorage.

## Getting Started

### Web App

```bash
cd web-app
npm install
npm run dev
```

Open the printed Vite URL, upload a PDF in the UI, and add your API key in Settings before chatting.

Available scripts:

- `npm run dev` — start the Vite dev server
- `npm run build` — build for production
- `npm run lint` — run ESLint
- `npm run preview` — preview the production build

### Desktop App

```bash
cd desktop-app
pip install -r requirements.txt
# create a .env file in desktop-app/ with: GOOGLE_API_KEY=your_key_here
streamlit run app.py
```

The desktop app uses Google Gemini (`gemini-2.5-flash`) and requires a `GOOGLE_API_KEY` in the environment. See `requirements.txt` for dependencies.

A bare CLI loop is also available via `python main.py` (loads `report.pdf`).

## Project Structure

```
├── web-app/                 # React browser app (recommended)
│   ├── src/
│   │   ├── App.jsx          # Main UI
│   │   ├── components/      # UI components (e.g. SettingsModal)
│   │   └── services/
│   │       ├── pdf.js       # PDF parsing & chunking
│   │       ├── db.js        # IndexedDB (Dexie) persistence
│   │       ├── retrieval.js # Local keyword-based retrieval
│   │       ├── llm.js       # Streaming completions per provider
│   │       └── storage.js   # Settings persistence (localStorage)
│   └── package.json
└── desktop-app/             # Streamlit + LangChain app
    ├── app.py               # Streamlit UI
    ├── main.py              # CLI loop
    └── src/
        ├── loader.py        # PDF loading
        ├── chunker.py       # Document chunking
        ├── vector_store.py  # Chroma vector store
        └── rag_chain.py     # Retrieval + Gemini QA chain
```

## How It Works

1. **Ingest**: A PDF is parsed into text and split into page-aware chunks.
2. **Index**: Chunks are stored/indexed locally — IndexedDB in the browser app, Chroma in the desktop app.
3. **Retrieve**: The user's question is matched against the indexed chunks to pull the most relevant context.
4. **Answer**: The context and question are sent to the chosen LLM, whose prompt requires citing page numbers — or stating clearly when an answer cannot be found in the provided notes.