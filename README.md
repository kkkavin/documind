# DocuMind AI

A privacy-first RAG (Retrieval-Augmented Generation) document assistant. Upload PDF notes or textbooks, index them, and ask questions with AI answers that cite the page numbers where the information was found.

## Features

- **Source-cited answers**: The assistant always states the page number(s) where it found the information.
- **PDF indexing**: PDFs are parsed, chunked, and embedded using sentence-transformers.
- **Vector store**: ChromaDB provides fast similarity search over the indexed chunks.
- **Google Gemini**: Answers are generated using Gemini (`gemini-2.5-flash`).

## Getting Started

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:

```
GOOGLE_API_KEY=your_key_here
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Or use the CLI loop:

```bash
python main.py
```

The CLI loop loads `report.pdf` by default.

## Project Structure

```
├── app.py               # Streamlit UI
├── main.py              # CLI loop
├── requirements.txt     # Python dependencies
└── src/
    ├── loader.py        # PDF loading
    ├── chunker.py       # Document chunking
    ├── vector_store.py  # Chroma vector store
    └── rag_chain.py     # Retrieval + Gemini QA chain
```
