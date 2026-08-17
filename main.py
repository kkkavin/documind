"""DocuMind CLI — index a notes folder and chat with it from the terminal.

Examples:
    python main.py --folder C:/MyNotes --download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf
    python main.py --folder C:/MyNotes --model ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from src.multi_loader import load_folder
from src.model_manager import create_llama, download_model, list_cached_models
from src.rag_chain import generate
from src.vector_store import index_folder, query_vector_store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocuMind local RAG assistant (CLI)")
    parser.add_argument("--folder", help="Absolute path to the notes folder")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a local .gguf model (defaults to the first cached model)",
    )
    parser.add_argument(
        "--download",
        nargs=2,
        metavar=("REPO", "FILENAME"),
        help="Download a GGUF model before chatting",
    )
    parser.add_argument("--index-only", action="store_true", help="Index the folder and exit")
    parser.add_argument("--list-models", action="store_true", help="List locally cached models")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window in tokens")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers (0 = CPU only)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved sources")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max answer tokens")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_models:
        models = list_cached_models()
        if not models:
            print("No cached models. Use --download to fetch one.")
            return 0
        for m in models:
            print(m)
        return 0

    if args.download:
        repo, filename = args.download
        with tqdm(unit="MB", unit_scale=True, desc=f"Downloading {filename}") as bar:
            def _cb(done: int, total: int) -> None:
                bar.total = total
                bar.n = done
                bar.refresh()

            path = download_model(repo, filename, progress_callback=_cb)
        print(f"Saved to {path}")

    if args.folder:
        print(f"Indexing {args.folder} …")
        summary = index_folder(
            args.folder,
            progress_callback=lambda action, cur, total, name, detail: print(
                f"  [{cur}/{total}] {action}: {name} ({detail})"
            ),
        )
        print(
            f"Indexed: {summary['added']} added, {summary['modified']} updated, "
            f"{summary['removed']} removed, {len(summary['errors'])} errors"
        )
        for err in summary["errors"]:
            print(f"  ⚠ {err}")
        if args.index_only:
            return 0
    else:
        print("No --folder given; skipping indexing.", file=sys.stderr)

    model_path = args.model
    if not model_path:
        cached = list_cached_models()
        if not cached:
            print("No local model found. Use --download or --model.", file=sys.stderr)
            return 1
        model_path = f"./models/{cached[0]}"
        print(f"Using cached model: {model_path}")

    print("\nStarting llama.cpp (first run downloads the binaries)…")
    with tqdm(unit="MB", unit_scale=True, desc="llama.cpp binaries") as bar:
        def _cb(done: int, total: int) -> None:
            bar.total = total
            bar.n = done
            bar.refresh()

        llm = create_llama(
            model_path=model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            progress_callback=_cb,
        )
    print("\nDocuMind ready. Type 'exit' to quit.\n")
    history: list[dict] = []
    try:
        while True:
            try:
                query = input(">> ")
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if query.strip().lower() == "exit":
                print("Bye!")
                break

            history.append({"role": "user", "content": query})
            sources, stream = generate(
                query,
                llm,
                store=None,
                folder=args.folder or ".",
                k=args.top_k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                history=history[:-1],
            )
            answer = "".join(stream)
            print()
            print(answer)
            print("\nSources:")
            for doc, score in sources:
                location = doc.metadata.get("page") or doc.metadata.get("line") or "?"
                print(f"  • {doc.metadata.get('file_name', '?')} · p.{location} ({score:.0%})")
            history.append({"role": "assistant", "content": answer})
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())