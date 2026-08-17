# DocuMind — Local Study Assistant

A privacy-first, fully local RAG (Retrieval-Augmented Generation) assistant.
Point it at a folder of study notes and ask questions. Answers come with
**source citations** (file + page/line), and **nothing ever leaves your
machine** — no cloud, no API keys.

## Features

- **100% local**: small GGUF language models (Qwen2.5 / Llama 3.2) run via
  official **llama.cpp binaries** (auto-downloaded once into `llama_bin/`),
  CPU-first by default. No Python C++ compilation needed — works on any
  Python version, including 3.14.
- **Dynamic model download**: fetch any GGUF from Hugging Face straight from
  the sidebar (presets or custom repo + filename), with a live progress bar.
- **Multi-format folders**: recursively indexes PDFs (PyMuPDF), DOCX, PPTX,
  spreadsheets (CSV, XLSX, XLS), TXT, Markdown and source code
  (`py`, `js`, `json`, `cpp`, `html`).
- **Incremental indexing**: only new/changed files are re-embedded; deleted
  files are purged automatically (per-folder state sidecar).
- **Source citations**: every answer shows the file, page/line, excerpt and
  match score of each retrieved chunk.
- **Streaming answers** with context-window, GPU-layer, top-K, temperature
  and max-token controls.

## Getting Started

```bash
pip install -r requirements.txt
```

No compiler or special wheels needed: the first time a chat starts, the app
downloads the official prebuilt `llama-server` CPU build from the llama.cpp
GitHub releases into `llama_bin/` (one-time, ~20 MB).

Run the app:

```bash
python -m streamlit run app.py
```

1. **Model** tab — pick a preset (or custom repo), click **Download model**,
   then select it as the active model.
2. **Notes folder** tab — paste a folder path (or **Browse…**), click
   **Index folder**.
3. Ask questions; answers stream in with a *Source Citations* expander.

## CLI

```bash
# List cached models
python main.py --list-models

# Download a model, then index a folder
python main.py --folder C:/MyNotes --download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf

# Index only
python main.py --folder C:/MyNotes --index-only

# Interactive REPL
python main.py --folder C:/MyNotes --model ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

## Project Structure

```
├── app.py                 # Streamlit UI
├── main.py                # CLI (download / index / chat)
├── requirements.txt       # Python dependencies
├── models/                # downloaded GGUF files (gitignored)
├── llama_bin/             # llama.cpp binaries (auto-downloaded, gitignored)
├── chroma_db/             # vector index (gitignored)
└── src/
    ├── model_manager.py   # HF GGUF download + llama-server inference
    ├── multi_loader.py    # folder scan + multi-format parsing
    ├── vector_store.py    # incremental persistent Chroma (all-MiniLM-L6-v2)
    └── rag_chain.py       # retrieval + prompt + streaming generation
```