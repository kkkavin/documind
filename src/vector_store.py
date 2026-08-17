"""Persistent incremental vector store for the DocuMind assistant.

Replaces the old bge-small-en-v1.5 Chroma store. Uses
``sentence-transformers/all-MiniLM-L6-v2`` embeddings and a cosine HNSW space
so every returned score is a similarity in [0, 1] (``score = 1 - distance``).

Indexing is incremental: a sidecar state file tracks each file's mtime/size,
so re-indexing a folder only embeds new or modified files. Collection names
are namespaced by folder hash + embedding model, which isolates different
folders and makes old bge collections obsolete (they are simply not queried).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.multi_loader import chunk_documents, load_file, scan_files

# ChromaDB will refuse a persist directory that does not exist.
CHROMA_DIR = Path("./chroma_db")
CHROMA_DIR.mkdir(exist_ok=True, parents=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Used to name the collection; sanitized below because Chroma names dislike '/'.
EMBEDDING_TAG = EMBEDDING_MODEL.split("/")[-1]

# Shared across requests so embeddings are computed exactly once per session.
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _folder_digest(folder: str | Path) -> str:
    """Stable short hash of a folder's absolute path (namespacing key)."""
    raw = str(Path(folder).resolve()).lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _state_path(folder: str | Path) -> Path:
    """Sidecar JSON tracking {source: [mtime, size]} for incremental indexing."""
    return CHROMA_DIR / f"index_state_{_folder_digest(folder)}.json"


def _load_state(folder: str | Path) -> dict:
    path = _state_path(folder)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _save_state(folder: str | Path, state: dict) -> None:
    _state_path(folder).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _open_store(folder: str | Path) -> Chroma:
    """Open (creating if needed) the Chroma collection for ``folder``."""
    collection_name = f"documind_{_folder_digest(folder)}_{EMBEDDING_TAG}"
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _delete_by_source(store: Chroma, source: str) -> None:
    """Remove every chunk belonging to ``source``; falls back to id-based delete."""
    try:
        store.delete(where={"source": source})
        return
    except (TypeError, ValueError):
        pass
    try:
        ids = store.get(where={"source": source})["ids"]
        if ids:
            store.delete(ids=ids)
    except Exception:  # noqa: BLE001 - the store may simply have nothing for this source
        pass


def index_folder(
    folder: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    progress_callback=None,
) -> dict:
    """Index (or update) every supported file under ``folder``.

    Only files whose mtime/size changed since the last run are re-embedded;
    deleted files are purged from the index. Returns a summary dict for the UI.

    ``progress_callback(action, current, total, name, detail)`` fires per file
    with ``action`` in {"index", "delete", "error", "skip"}.
    """
    files = scan_files(folder)
    state = _load_state(folder)
    current_sources = {str(p) for p in files}

    summary = {"total_files": len(files), "added": 0, "modified": 0, "removed": 0, "errors": []}

    to_delete = sorted(set(state) - current_sources)
    for i, source in enumerate(to_delete, start=1):
        if progress_callback is not None:
            progress_callback("delete", i, len(to_delete), Path(source).name, "removed from index")
        try:
            _delete_by_source(_open_store(folder), source)
            state.pop(source, None)
            summary["removed"] += 1
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"{Path(source).name}: {exc}")

    todo: list[Path] = []
    for path in files:
        try:
            stat = path.stat()
            key = str(path)
            cached = state.get(key)
            if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                continue
            todo.append(path)
        except OSError:
            continue

    store = _open_store(folder)
    for i, path in enumerate(todo, start=1):
        try:
            docs = chunk_documents(load_file(path), chunk_size, chunk_overlap)
            source = str(path)
            # A modified file keeps stale chunks unless we clear them first.
            if source in state:
                _delete_by_source(store, source)
                summary["modified"] += 1
                action = "modified"
            else:
                summary["added"] += 1
                action = "index"
            if docs:
                store.add_documents(docs)
            stat = path.stat()
            state[source] = [stat.st_mtime, stat.st_size]
            if progress_callback is not None:
                progress_callback(action, i, len(todo), path.name, f"{len(docs)} chunks")
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort indexing
            summary["errors"].append(f"{path.name}: {exc}")
            if progress_callback is not None:
                progress_callback("error", i, len(todo), path.name, str(exc))

    _save_state(folder, state)
    return summary


def query_vector_store(folder: str | Path, query: str, k: int = 3) -> list[tuple]:
    """Return up to ``k`` (document, score) pairs for ``query``.

    The store uses cosine distance, so similarity = 1 - distance; results are
    sorted best-first so the UI can display confidence percentages.
    """
    store = _open_store(folder)
    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception:  # noqa: BLE001 - empty/never-indexed collections raise
        return []
    scored = [(doc, round(1.0 - distance, 4)) for doc, distance in results]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored