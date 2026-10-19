import os
from pathlib import Path

import streamlit as st

# set_page_config MUST be the first Streamlit command in the script.
st.set_page_config(page_title="PDF Research Assistant", page_icon="📚", layout="wide")


def _bridge_secrets() -> None:
    """Copy Streamlit Cloud secrets into env so config.py works unchanged.

    Only touched when a secrets file actually exists, to avoid the noisy
    'no secrets file' notice during local runs that use a plain .env.
    """
    candidates = [Path.home() / ".streamlit" / "secrets.toml",
                  Path(".streamlit") / "secrets.toml"]
    if not any(p.exists() for p in candidates):
        return
    try:
        for key in ("GOOGLE_API_KEY", "APP_PASSWORD", "GEMINI_CHAT_MODEL",
                    "GEMINI_EMBEDDING_MODEL", "DATA_DIR"):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


_bridge_secrets()

import config
from auth import require_login
from document_store import DocumentStore
from rag_pipeline import (
    build_prompt_from_docs,
    sources_from_docs,
    stream_answer,
    summarize_documents,
)

# Static styling only — no dynamic HTML is injected (avoids XSS from PDF text).
st.markdown(
    """
    <style>
      .stApp { font-family: 'Inter', sans-serif; }
      .source-pill {
        display: inline-block; padding: 2px 10px; margin: 2px 6px 2px 0;
        border-radius: 999px; background: #eef2ff; color: #3730a3;
        font-size: 0.8rem; border: 1px solid #c7d2fe;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

require_login()  # stops here if a password is configured and not yet entered

# --- Session state --------------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = DocumentStore.load()  # restore persisted docs
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("summaries", {})

store: DocumentStore = st.session_state.store


def render_sources(sources) -> None:
    if not sources:
        return
    pills = "".join(
        f"<span class='source-pill'>{s['source']} · p.{s['page']}</span>"
        for s in sources
    )
    st.markdown("**Sources:** " + pills, unsafe_allow_html=True)


# --- Sidebar: document manager -------------------------------------------
st.sidebar.title("📚 PDF Research Assistant")

if not config.has_api_key():
    st.sidebar.error("GOOGLE_API_KEY is missing. Add it to .env / secrets.")

st.sidebar.subheader("Add documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)
use_ocr = st.sidebar.checkbox(
    "OCR scanned pages", value=config.OCR_ENABLED,
    help="Read image-only pages with Gemini vision (uses extra API calls).",
)

if st.sidebar.button("➕ Add to library", use_container_width=True):
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF first.")
    else:
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in uploaded_files:
            try:
                dest = config.UPLOAD_DIR / pdf.name
                dest.write_bytes(pdf.getbuffer())
                status = st.sidebar.status(f"Indexing {pdf.name}…", expanded=False)

                def ocr_progress(cur, total, _name=pdf.name):
                    status.update(label=f"OCR {_name}: page {cur}/{total}")

                info = store.add_pdf(
                    str(dest), pdf.name, enable_ocr=use_ocr, ocr_progress=ocr_progress
                )
                st.session_state.summaries.pop(pdf.name, None)
                note = f" ({info['ocr_pages']} page(s) via OCR)" if info.get("ocr_pages") else ""
                status.update(label=f"Added {pdf.name}: {info['chunks']} chunks{note}",
                              state="complete")
            except Exception as error:
                st.sidebar.error(f"{pdf.name}: {error}")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Library")
docs = store.list_documents()
if not docs:
    st.sidebar.caption("No documents yet. Upload a PDF above.")
else:
    for doc in docs:
        col_a, col_b = st.sidebar.columns([0.8, 0.2])
        pages = doc.get("pages", "?")
        ocr = f" · {doc['ocr_pages']} OCR" if doc.get("ocr_pages") else ""
        col_a.markdown(f"**{doc['name']}**  \n{doc.get('chunks','?')} chunks · "
                       f"{pages} pages{ocr}")
        if col_b.button("🗑️", key=f"del_{doc['name']}", help="Remove"):
            store.remove_document(doc["name"])
            st.session_state.summaries.pop(doc["name"], None)
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Chat: `{config.CHAT_MODEL}`")
st.sidebar.caption(f"Embeddings: `{config.EMBEDDING_MODEL}`")


# --- Main -----------------------------------------------------------------
st.title("PDF Research Assistant")
st.write("Ask questions grounded in your PDFs (with citations), or summarize an "
         "entire document — even a full book — without hallucination.")

if store.is_empty():
    st.info("👈 Add one or more PDFs in the sidebar to get started.")
    st.stop()

ask_tab, summary_tab = st.tabs(["💬 Ask", "📝 Summarize"])

# --- Ask tab --------------------------------------------------------------
with ask_tab:
    names = store.names()
    scope = st.multiselect(
        "Search within", names, default=names,
        help="Limit the search to specific documents (default: all).",
    )

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            render_sources(turn.get("sources"))

    question = st.chat_input("Ask a question about your PDFs…")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                docs = store.search(question, sources=scope or None)
                if not docs:
                    st.warning("No matching content found in the selected documents.")
                else:
                    prompt = build_prompt_from_docs(
                        question, docs, st.session_state.chat_history
                    )
                    answer = st.write_stream(stream_answer(prompt))
                    sources = sources_from_docs(docs)
                    render_sources(sources)
                    st.session_state.chat_history.append(
                        {"question": question, "answer": answer, "sources": sources}
                    )
                    st.session_state.chat_history = st.session_state.chat_history[-20:]
            except Exception as error:
                st.error(f"Could not answer the question: {error}")

    if st.session_state.chat_history and st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- Summarize tab --------------------------------------------------------
with summary_tab:
    names = store.names()
    options = (["All documents"] + names) if len(names) > 1 else names
    choice = st.selectbox("Choose what to summarize", options)

    if choice == "All documents":
        selected_docs, title = store.all_documents(), "All documents"
    else:
        selected_docs, title = store.documents_for(choice), choice

    est_chars = sum(len(d.page_content) for d in selected_docs)
    st.caption(f"~{len(selected_docs)} chunks · ~{est_chars:,} characters. "
               "Long documents make multiple model calls and may take a while.")

    if st.button("✨ Generate summary", use_container_width=True):
        status = st.status("Summarizing…", expanded=True)
        bar = status.progress(0.0)

        def on_progress(stage, current, total):
            labels = {"map": "Reading & summarizing sections",
                      "reduce": "Combining section summaries",
                      "final": "Writing final summary"}
            fraction = 0.99 if stage == "final" else current / max(total, 1)
            bar.progress(min(fraction, 1.0),
                         text=f"{labels.get(stage, stage)} ({current}/{total})")

        try:
            summary = summarize_documents(selected_docs, title=title, progress=on_progress)
            st.session_state.summaries[choice] = summary
            status.update(label="Summary ready", state="complete", expanded=False)
        except Exception as error:
            status.update(label="Summarization failed", state="error")
            st.error(f"Could not summarize: {error}")

    if choice in st.session_state.summaries:
        summary = st.session_state.summaries[choice]
        st.markdown("### Summary")
        st.markdown(summary)
        st.download_button(
            "⬇️ Download summary (.md)",
            data=summary,
            file_name=f"{Path(str(choice)).stem or 'summary'}_summary.md",
            mime="text/markdown",
        )
