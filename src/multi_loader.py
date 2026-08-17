"""Multi-format directory scanner and loader.

Replaces the old single-PDF ``loader.py``/``chunker.py`` pair. Recursively
scans a folder and parses PDFs, Word documents, spreadsheets, presentations,
plain text, markdown and common source code files into LangChain ``Document``
objects with rich metadata (``source``, ``file_name``, ``page``/``line``,
``sheet`` or ``slide``).

Includes a Streamlit-safe native folder picker: ``tkinter`` runs in a
separate subprocess so it cannot freeze the Streamlit script thread, and the
helper degrades gracefully (returns ``None``) on headless systems.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# File types the assistant understands and how to parse them.
TEXT_EXTENSIONS = {".txt", ".md"}
CODE_EXTENSIONS = {".py", ".js", ".json", ".cpp", ".html"}
SPREADSHEET_EXTENSIONS = {".csv", ".xlsx", ".xls"}
PRESENTATION_EXTENSIONS = {".pptx"}
SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | CODE_EXTENSIONS
    | SPREADSHEET_EXTENSIONS
    | PRESENTATION_EXTENSIONS
    | {".pdf", ".docx"}
)

# Skip anything larger than this (e.g. media-heavy PDFs or binary junk).
MAX_FILE_BYTES = 50 * 1024 * 1024

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def browse_folder() -> str | None:
    """Open a native directory picker in a subprocess; return the path or ``None``.

    Works on Windows/Linux/macOS desktops. On headless servers (or if tkinter
    is missing) it returns ``None`` so the caller can fall back to manual input.
    """
    code = (
        "import tkinter as tk;"
        "from tkinter import filedialog;"
        "root = tk.Tk();"
        "root.withdraw();"
        "root.attributes('-topmost', True);"
        "root.update_idletasks();"
        "print(filedialog.askdirectory(), end='')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
        path = result.stdout.strip()
        return path or None
    except Exception as exc:  # noqa: BLE001 - any failure just disables the button
        warnings.warn(f"Native folder dialog unavailable: {exc}")
        return None


def scan_files(folder: str | Path) -> list[Path]:
    """Recursively list supported files under ``folder``, sorted and size-guarded."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a valid directory.")

    files: list[Path] = []
    for dirpath, dirnames, filenames in os_walk(root):
        # Skip obvious junk directories.
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith((".", "_")) and d not in {"node_modules", "venv", "__pycache__"}
        ]
        for name in sorted(filenames):
            path = Path(dirpath) / name
            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)

    return files


def os_walk(root: Path):
    """Thin wrapper around os.walk for testability (imported locally to keep the header clean)."""
    import os

    return os.walk(root)


def _require_dep(module: str, pip: str):
    """Import ``module`` or raise a clear, actionable install message."""
    try:
        return __import__(module)
    except ImportError as exc:
        raise RuntimeError(
            f"'{pip}' is not installed — run: pip install {pip}"
        ) from exc


def load_pdf(path: Path) -> list[Document]:
    """Extract per-page text from a PDF with PyMuPDF."""
    pymupdf = _require_dep("pymupdf", "pymupdf")

    docs: list[Document] = []
    try:
        with pymupdf.open(path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "page": page_num,
                        },
                    )
                )
    except Exception as exc:  # noqa: BLE001 - corrupt PDFs should not abort indexing
        raise RuntimeError(f"Failed to parse PDF '{path.name}': {exc}") from exc
    return docs


def load_docx(path: Path) -> list[Document]:
    """Extract paragraphs and table text from a .docx file."""
    docx = _require_dep("docx", "python-docx")

    docs: list[Document] = []
    try:
        document = docx.Document(str(path))
        parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                parts.append(" | ".join(cells).strip())
        text = "\n".join(p for p in parts if p)
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "line": 1,
                    },
                )
            )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse DOCX '{path.name}': {exc}") from exc
    return docs


def load_csv(path: Path) -> list[Document]:
    """Read a CSV file; cells are tab-separated and rows newline-separated."""
    import csv

    rows: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.reader(fh):
                if row:
                    rows.append("\t".join(cell.strip() for cell in row))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse CSV '{path.name}': {exc}") from exc

    text = "\n".join(rows).strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "line": 1,
            },
        )
    ]


def load_xlsx(path: Path) -> list[Document]:
    """Extract text from each worksheet of an .xlsx file (read-only mode)."""
    openpyxl = _require_dep("openpyxl", "openpyxl")

    docs: list[Document] = []
    try:
        workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            rows = [
                "\t".join(
                    str(cell).strip() for cell in row if cell is not None and str(cell).strip()
                )
                for row in sheet.iter_rows(values_only=True)
                if any(cell is not None and str(cell).strip() for cell in row)
            ]
            text = "\n".join(rows).strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "sheet": sheet.title,
                        },
                    )
                )
        workbook.close()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse XLSX '{path.name}': {exc}") from exc
    return docs


def load_xls(path: Path) -> list[Document]:
    """Extract text from each sheet of a legacy .xls file (xlrd)."""
    xlrd = _require_dep("xlrd", "xlrd")

    docs: list[Document] = []
    try:
        workbook = xlrd.open_workbook(str(path))
        for sheet in workbook.sheets():
            rows = []
            for row_idx in range(sheet.nrows):
                values = [
                    str(sheet.cell_value(row_idx, col_idx)).strip()
                    for col_idx in range(sheet.ncols)
                ]
                if any(values):
                    rows.append("\t".join(values))
            text = "\n".join(rows).strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "sheet": sheet.name,
                        },
                    )
                )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse XLS '{path.name}': {exc}") from exc
    return docs


def load_pptx(path: Path) -> list[Document]:
    """Extract text from each slide of a .pptx presentation (shapes + tables)."""
    pptx = _require_dep("pptx", "python-pptx")

    def _shape_text(shape) -> str:
        parts: list[str] = []
        if getattr(shape, "has_text_frame", False) and shape.text_frame:
            for paragraph in shape.text_frame.paragraphs:
                text = "".join(run.text for run in paragraph.runs).strip()
                if text:
                    parts.append(text)
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                parts.append(" | ".join(cells).strip())
        if getattr(shape, "shape_type", None) is not None:
            try:
                for sub in shape.shapes:
                    parts.append(_shape_text(sub))
            except Exception:  # noqa: BLE001 - group shape traversal is best-effort
                pass
        return "\n".join(parts).strip()

    docs: list[Document] = []
    try:
        presentation = pptx.Presentation(str(path))
        for slide_idx, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                text = _shape_text(shape)
                if text:
                    parts.append(text)
            text = "\n".join(parts).strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "slide": slide_idx,
                        },
                    )
                )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse PPTX '{path.name}': {exc}") from exc
    return docs


def load_text_file(path: Path) -> list[Document]:
    """Read a plain text / markdown / source code file as one document."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read '{path.name}': {exc}") from exc

    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "line": 1,
            },
        )
    ]


def load_file(path: Path) -> list[Document]:
    """Dispatch a single file to the right parser based on its extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext in SPREADSHEET_EXTENSIONS:
        if ext == ".csv":
            return load_csv(path)
        if ext == ".xlsx":
            return load_xlsx(path)
        return load_xls(path)
    if ext in PRESENTATION_EXTENSIONS:
        return load_pptx(path)
    if ext in TEXT_EXTENSIONS or ext in CODE_EXTENSIONS:
        return load_text_file(path)
    return []


def chunk_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks, tagging each with its index.

    The splitter preserves the per-page metadata of PDF documents, so every
    chunk keeps its ``page`` number. Text/code chunks keep ``line=1`` and gain
    a ``chunk`` index.
    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    return chunks


def load_folder(
    folder: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    progress_callback=None,
) -> tuple[list[Document], dict]:
    """Scan a folder, parse every supported file and return all chunks.

    ``progress_callback(action, current, total, name, detail)`` is invoked per
    file — used by the Streamlit UI to render a status bar.

    Returns ``(chunks, summary)`` where summary carries counts for the UI.
    """
    files = scan_files(folder)
    total = len(files)
    summary = {"total_files": total, "errors": []}
    all_docs: list[Document] = []

    for i, path in enumerate(files, start=1):
        try:
            docs = load_file(path)
            all_docs.extend(docs)
            action = "load"
        except Exception as exc:  # noqa: BLE001 - one bad file should not kill indexing
            action = "error"
            summary["errors"].append(f"{path.name}: {exc}")
            docs = []
        if progress_callback is not None:
            progress_callback(action, i, total, path.name, f"{len(docs)} pages/sections")

    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)
    summary["chunks"] = len(chunks)
    return chunks, summary