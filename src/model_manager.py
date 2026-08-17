"""Dynamic Hugging Face GGUF model downloader and llama.cpp inference manager.

Responsibilities:
- Download GGUF models from Hugging Face into a local ``./models/`` directory
  with byte-accurate progress reporting (tqdm does not render inside Streamlit,
  so we poll the on-disk bytes against the exact file size from the HF API).
- List locally cached models and the files of any HF repo (to avoid mistyping
  filenames).
- Manage a ``llama-server`` subprocess (official llama.cpp prebuilt binaries)
  for chat inference: start it on a free local port, wait until healthy, and
  stream completions over its OpenAI-compatible HTTP API. This avoids the
  ``llama-cpp-python`` build requirement entirely (works on any Python
  version, including 3.14, with no C++ toolchain).

This module is UI-agnostic (no Streamlit imports) so it can be shared between
``app.py`` and the ``main.py`` CLI. If the repo is private/gated, ``huggingface_hub``
automatically picks up an ``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Default location for downloaded GGUF models.
MODELS_DIR = Path(os.environ.get("DOCUMIND_MODELS_DIR", "./models"))

# Where the llama.cpp binaries are kept (auto-downloaded once).
LLAMA_BIN_DIR = Path(os.environ.get("DOCUMIND_LLAMA_DIR", "./llama_bin"))
LLAMA_SERVER_EXE = LLAMA_BIN_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server")

# Fallback release tag if the GitHub API is unreachable.
LLAMA_CPP_PINNED_TAG = "b10472"

# Recommended lightweight models for CPU-first laptops / low-end PCs.
# ``filename`` must match a file in the repo — verified per preset at download time.
PRESET_MODELS = [
    {
        "label": "Qwen2.5 0.5B Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size": "~400 MB",
    },
    {
        "label": "Qwen2.5 1.5B Instruct (Q4_K_M)",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size": "~1.1 GB",
    },
    {
        "label": "Llama 3.2 1B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size": "~800 MB",
    },
    {
        "label": "Llama 3.2 3B Instruct (Q4_K_M)",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size": "~2.0 GB",
    },
]


def ensure_models_dir() -> Path:
    """Create and return the local models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def list_cached_models() -> list[str]:
    """Return names of all ``.gguf`` files already present in ``./models``."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("*.gguf"))


def list_repo_files(repo_id: str) -> list[str]:
    """List the files of a model repo (used by the UI to avoid typos)."""
    if not repo_id.strip():
        raise ValueError("Please enter a Hugging Face repo ID first.")
    return list(HfApi().list_repo_files(repo_id.strip(), repo_type="model"))


def get_repo_file_size(repo_id: str, filename: str) -> int:
    """Return the exact byte size of ``filename`` inside ``repo_id``.

    Raises ``FileNotFoundError`` with a helpful message if the file is not in
    the repo, and ``ValueError`` if the repo is gated so sizes are hidden.
    """
    info = HfApi().model_info(repo_id.strip(), files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == filename:
            if sibling.size is None:
                raise ValueError(
                    f"Could not determine the size of '{filename}' — the repo may be "
                    "gated. Set HF_TOKEN or use a public model."
                )
            return sibling.size
    raise FileNotFoundError(
        f"'{filename}' was not found in '{repo_id}'. Use the 'List repo files' "
        "button to see the available filenames."
    )


def _bytes_on_disk(repo_id: str, filename: str, local_dir: Path) -> int:
    """Best-effort byte count of the download, looking at the target and the HF cache."""
    target = local_dir / filename
    if target.exists():
        try:
            return target.stat().st_size
        except OSError:
            pass

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    if model_dir.exists():
        for candidate in model_dir.rglob(filename):
            try:
                return candidate.stat().st_size
            except OSError:
                continue
    return 0


def download_model(
    repo_id: str,
    filename: str,
    local_dir: Path | None = None,
    progress_callback=None,
) -> Path:
    """Download a GGUF file into ``local_dir`` (default ``./models``).

    ``hf_hub_download`` runs on a daemon thread while the caller polls the
    on-disk bytes so the UI can render a live progress bar. ``progress_callback``
    is invoked with ``(downloaded_bytes, total_bytes)``.
    """
    local_dir = local_dir or ensure_models_dir()
    local_dir.mkdir(parents=True, exist_ok=True)

    total = get_repo_file_size(repo_id, filename)
    target = local_dir / filename
    if target.exists() and target.stat().st_size >= total:
        return target

    result: dict = {"path": None, "error": None}

    def _worker() -> None:
        try:
            result["path"] = hf_hub_download(
                repo_id=repo_id.strip(),
                filename=filename,
                local_dir=str(local_dir),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            result["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while thread.is_alive():
        done = _bytes_on_disk(repo_id, filename, local_dir)
        if progress_callback is not None:
            progress_callback(min(done, total), total)
        time.sleep(0.2)

    if result["error"] is not None:
        raise result["error"]

    # Very old huggingface_hub versions only write to the cache dir; make sure
    # the model actually lands in ./models.
    source = Path(result["path"])
    if source != target:
        shutil.copy2(source, target)

    if progress_callback is not None:
        progress_callback(total, total)
    return target


# --------------------------------------------------------------------------
# llama.cpp binaries (llama-server)
# --------------------------------------------------------------------------

def _latest_llama_cpp_tag() -> str:
    """Return the latest llama.cpp release tag via the GitHub API."""
    try:
        with urllib.request.urlopen(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            timeout=15,
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["tag_name"]
    except Exception:  # noqa: BLE001 - fall back to the pinned tag
        return LLAMA_CPP_PINNED_TAG


def _find_win_cpu_asset(assets: list[dict]) -> str | None:
    """Pick the best Windows x64 CPU asset: prefer cpu, then avx2, then avx."""
    for kind in ("cpu", "avx2", "avx"):
        for asset in assets:
            name = asset.get("name", "")
            if name.endswith(f"bin-win-{kind}-x64.zip"):
                return name
    return None


def ensure_llama_server(progress_callback=None) -> Path:
    """Return the path to ``llama-server``, downloading it once if needed.

    Downloads the official prebuilt CPU build from the llama.cpp GitHub
    releases and extracts ``llama-server.exe`` into ``./llama_bin/``.
    ``progress_callback(done, total)`` is invoked during the zip download.
    """
    if LLAMA_SERVER_EXE.exists():
        return LLAMA_SERVER_EXE

    LLAMA_BIN_DIR.mkdir(parents=True, exist_ok=True)
    tag = _latest_llama_cpp_tag()

    try:
        with urllib.request.urlopen(
            f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}",
            timeout=15,
        ) as resp:
            release = json.loads(resp.read().decode("utf-8"))
        asset_name = _find_win_cpu_asset(release.get("assets", []))
    except Exception:  # noqa: BLE001
        asset_name = None

    if not asset_name:
        raise RuntimeError(
            f"Could not find a Windows CPU build for llama.cpp release '{tag}'. "
            f"Download 'llama-{tag}-bin-win-cpu-x64.zip' from "
            "https://github.com/ggml-org/llama.cpp/releases manually and place "
            f"llama-server.exe in '{LLAMA_BIN_DIR}'."
        )

    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset_name}"
    zip_path = LLAMA_BIN_DIR / asset_name

    with urllib.request.urlopen(url, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with zip_path.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress_callback is not None and total:
                    progress_callback(done, total)

    with zipfile.ZipFile(zip_path) as zf:
        # Extract everything: llama-server.exe is a small launcher that needs
        # its companion DLLs (ggml-*, libomp, ...) next to it at runtime.
        zf.extractall(LLAMA_BIN_DIR)
    zip_path.unlink(missing_ok=True)

    if not LLAMA_SERVER_EXE.exists():
        raise RuntimeError("llama-server.exe was not found inside the downloaded archive.")
    return LLAMA_SERVER_EXE


def _free_port() -> int:
    """Ask the OS for a free TCP port on localhost."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --------------------------------------------------------------------------
# LlamaServer: llama-server subprocess wrapper
# --------------------------------------------------------------------------

class LlamaServer:
    """A managed ``llama-server`` subprocess exposing chat completions.

    The GGUF's baked-in chat template is used automatically by llama-server,
    so Qwen / Llama models get correct templating with zero extra work.
    """

    START_TIMEOUT = 180  # generous: model load + warmup on slow CPUs

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        progress_callback=None,
    ) -> None:
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.port = _free_port()
        self.process: subprocess.Popen | None = None
        self._stderr_tail: list[str] = []
        self._start(progress_callback)

    # -- lifecycle ---------------------------------------------------------

    def _start(self, progress_callback=None) -> None:
        exe = ensure_llama_server(progress_callback)
        cmd = [
            str(exe),
            "-m", self.model_path,
            "--ctx-size", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-ngl", str(self.n_gpu_layers),
            "--host", "127.0.0.1",
            "--port", str(self.port),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        # Drain stderr continuously: if the pipe buffer ever fills, the server
        # blocks on write and stops answering requests.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        deadline = time.time() + self.START_TIMEOUT
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup:\n{self._stderr()}"
                )
            if self._health():
                return
            time.sleep(0.5)
        raise RuntimeError(
            f"llama-server did not become ready within {self.START_TIMEOUT}s:\n{self._stderr()}"
        )

    def _drain_stderr(self) -> None:
        """Keep the stderr pipe drained and remember its recent lines."""
        stream = self.process.stderr if self.process is not None else None
        if stream is None:
            return
        for line in stream:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 50:
                self._stderr_tail.pop(0)

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/health", timeout=2
            ) as resp:
                return json.loads(resp.read().decode("utf-8")).get("status") == "ok"
        except Exception:  # noqa: BLE001 - server may still be starting
            return False

    def _stderr(self) -> str:
        lines = self._stderr_tail[-20:]
        return "\n".join(lines) if lines else "(no output captured)"

    def close(self) -> None:
        """Stop the server process (best-effort, idempotent)."""
        proc = self.process
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from __del__
            pass


def create_llama(
    model_path: str | Path,
    n_ctx: int = 2048,
    n_threads: int | None = None,
    n_gpu_layers: int = 0,
    verbose: bool = False,
    progress_callback=None,
) -> LlamaServer:
    """Start a llama-server for ``model_path`` tuned for consumer CPUs.

    Args:
        model_path: Path to a local ``.gguf`` file.
        n_ctx: Context window in tokens.
        n_threads: CPU threads; defaults to ``os.cpu_count()``.
        n_gpu_layers: Layers offloaded to GPU. 0 = pure CPU (the CPU build
            ignores this and warns if set above 0).
        verbose: Pass through llama.cpp verbosity.
        progress_callback: Called with ``(done, total)`` during the one-time
            llama.cpp binary download.
    """
    return LlamaServer(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        progress_callback=progress_callback,
    )


def stream_chat(server: LlamaServer, messages: list[dict], temperature: float = 0.7, max_tokens: int = 512):
    """Stream a chat completion from a running ``LlamaServer``.

    POSTs to the OpenAI-compatible ``/v1/chat/completions`` endpoint with
    ``stream=True`` and yields ``delta.content`` chunks as SSE events arrive.
    """
    payload = json.dumps(
        {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        response = urllib.request.urlopen(request, timeout=3600)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach llama-server ({exc}). "
            f"Server log:\n{server._stderr()}"
        ) from exc

    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                yield text