"""DocuMind — local, multi-format RAG assistant (Streamlit UI).

Run with:  streamlit run app.py
"""

import threading
import time

import streamlit as st

from src.multi_loader import browse_folder, load_folder
from src.model_manager import (
    PRESET_MODELS,
    create_llama,
    download_model,
    list_cached_models,
    list_repo_files,
)
from src.rag_chain import generate
from src.vector_store import index_folder

st.set_page_config(page_title="DocuMind — Local Study Assistant", page_icon="🧠")
st.title("🧠 DocuMind: Local Notes Assistant")

st.caption(
    "Everything runs on this machine — no cloud, no API keys. Pick a model, point at a "
    "folder of notes (PDF / DOCX / TXT / MD / code), and ask away."
)


# --------------------------------------------------------------------------
# Sidebar: model management
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Model")

    st.session_state.setdefault("repo_id", PRESET_MODELS[1]["repo_id"])

    model_tab = st.radio(
        "Source",
        ["Preset (recommended)", "Custom repo"],
        label_visibility="collapsed",
    )

    if model_tab == "Preset (recommended)":
        preset = st.selectbox(
            "Model preset",
            options=PRESET_MODELS,
            format_func=lambda m: f"{m['label']} ({m['size']})",
        )
        download_repo = preset["repo_id"]
        filename = preset["filename"]
        # Sync the repo field before the widget renders so it never drifts
        # from the selected preset.
        st.session_state["repo_id"] = download_repo
    else:
        filename = st.text_input("GGUF filename", value="", help="Tip: click 'List repo files'.")
        if st.button("List repo files"):
            try:
                files = list_repo_files(st.session_state["repo_id"])
                st.session_state["repo_files"] = files
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                st.error(f"Could not list files: {exc}")
        if "repo_files" in st.session_state:
            st.caption("Files in repo:")
            for f in st.session_state["repo_files"][:40]:
                st.code(f)
        download_repo = st.session_state["repo_id"]

    repo_id = st.text_input(
        "Hugging Face repo ID",
        key="repo_id",
        disabled=(model_tab == "Preset (recommended)"),
        help="e.g. Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    )

    if st.button("Download model", type="primary", disabled=not filename):
        progress_bar = st.progress(0.0, text="Starting download…")
        state = {"done": 0, "total": 0, "error": None, "ok": False}

        def _cb(done: int, total: int) -> None:
            state["done"], state["total"] = done, total

        def _work() -> None:
            try:
                download_model(download_repo, filename, progress_callback=_cb)
                state["ok"] = True
            except Exception as exc:  # noqa: BLE001 - shown to the user
                state["error"] = exc

        threading.Thread(target=_work, daemon=True).start()
        while not state["ok"] and state["error"] is None:
            if state["total"]:
                progress_bar.progress(
                    min(state["done"] / state["total"], 1.0),
                    text=f"Downloading {state['done'] / 1e6:.1f} / {state['total'] / 1e6:.1f} MB",
                )
            time.sleep(0.2)
        if state["error"] is not None:
            progress_bar.empty()
            st.error(f"Download failed: {state['error']}")
        else:
            progress_bar.progress(1.0, text="Download complete.")
            st.success(f"Saved to ./models/{filename}")

    cached = list_cached_models()
    if not cached:
        st.warning("No local models yet — download one above.")
    active_model = st.selectbox(
        "Active model (local .gguf)",
        options=cached or ["<none>"],
        help="Only GGUF files already present in ./models are shown.",
    )

    with st.expander("Runtime settings"):
        n_gpu_layers = st.slider("GPU layers (0 = CPU only)", 0, 99, 0, 1)
        n_ctx = st.slider("Context window (tokens)", 512, 8192, 2048, 512)

    st.divider()

    # ----------------------------------------------------------------------
    # Sidebar: folder & indexing
    # ----------------------------------------------------------------------
    st.header("2 · Notes folder")

    folder_path = st.text_input(
        "Folder path",
        key="folder_path",
        help="Absolute path to a folder of PDF/DOCX/TXT/MD/code files.",
    )
    def _pick_folder() -> None:
        picked = browse_folder()
        if picked:
            st.session_state["folder_path"] = picked
        else:
            st.session_state["picker_unavailable"] = True

    st.button("📂 Browse…", on_click=_pick_folder)
    if st.session_state.pop("picker_unavailable", False):
        st.info("Folder dialog unavailable — please type the path manually.")

    if st.button("Index folder", type="primary", disabled=not folder_path):
        status = st.status("Scanning folder…", expanded=True)
        progress = st.progress(0.0)

        def _cb(action: str, current: int, total: int, name: str, detail: str) -> None:
            status.write(f"`{action}` · {name} — {detail}")
            if total:
                progress.progress(current / total)

        summary = index_folder(folder_path, progress_callback=_cb)
        progress.empty()
        status.update(
            label=(
                f"Done: {summary['added']} added, {summary['modified']} updated, "
                f"{summary['removed']} removed, {summary['total_files']} files scanned."
            ),
            state="complete" if not summary["errors"] else "error",
            expanded=False,
        )
        if summary["errors"]:
            for err in summary["errors"][:10]:
                status.write(f"⚠ {err}")
        st.toast("Notes indexed — you can ask questions now.")

    # ----------------------------------------------------------------------
    # Sidebar: generation parameters
    # ----------------------------------------------------------------------
    st.header("3 · Generation")
    top_k = st.slider("Top-K sources", 1, 10, 3, 1)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max answer tokens", 64, 2048, 512, 64)


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not active_model or active_model == "<none>":
    st.info("⬅️ Download a model in the sidebar to get started.")
elif not folder_path:
    st.info("⬅️ Point the assistant at a notes folder and click **Index folder**.")
elif prompt := st.chat_input("Ask a question about your notes…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        sources = []
        answer = ""
        try:
            server_key = (active_model, n_ctx, n_gpu_layers)
            if st.session_state.get("server_key") != server_key:
                if "server" in st.session_state:
                    st.session_state.server.close()
                with st.status("Starting local model server…", expanded=True) as status:
                    progress_bar = st.progress(0.0)

                    def _cb(done: int, total: int) -> None:
                        if total:
                            progress_bar.progress(min(done / total, 1.0))

                    st.session_state.server = create_llama(
                        model_path=f"./models/{active_model}",
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        progress_callback=_cb,
                    )
                    status.update(
                        label="Model server ready.",
                        state="complete",
                        expanded=False,
                    )
                st.session_state.server_key = server_key

            sources, stream = generate(
                prompt,
                st.session_state.server,
                store=None,
                folder=folder_path,
                k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                history=st.session_state.messages[:-1],
            )
            answer = placeholder.write_stream(stream)
        except Exception as exc:  # noqa: BLE001 - any failure is shown inline
            placeholder.error(f"Generation failed: {exc}")
            answer = ""

        if sources:
            with st.expander(f"Source Citations ({len(sources)})"):
                for doc, score in sources:
                    location = doc.metadata.get("page") or doc.metadata.get("line") or "?"
                    st.markdown(
                        f"**{doc.metadata.get('file_name', '?')} · p.{location}** "
                        f"— match {score:.0%}"
                    )
                    st.text(doc.page_content[:400])

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})