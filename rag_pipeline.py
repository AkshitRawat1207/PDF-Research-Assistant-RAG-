"""RAG pipeline: PDF loading, embeddings/FAISS, grounded Q&A, and
hallucination-resistant whole-document summarization that scales to books.
"""

import os
import time
import logging
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Progress callback signature: (stage: str, current: int, total: int) -> None
ProgressFn = Callable[[str, int, int], None]


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    config.require_api_key()
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)


def get_llm(temperature: Optional[float] = None) -> ChatGoogleGenerativeAI:
    config.require_api_key()
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=config.TEMPERATURE if temperature is None else temperature,
        timeout=config.REQUEST_TIMEOUT,
        max_retries=config.MAX_RETRIES,
    )


# --------------------------------------------------------------------------
# OCR fallback for scanned / image-only pages (PyMuPDF render -> Gemini vision)
# --------------------------------------------------------------------------
_OCR_INSTRUCTION = (
    "You are an OCR engine. Extract ALL text from this document page image "
    "verbatim, preserving reading order. Output only the extracted text with no "
    "commentary. If the page contains no readable text, output nothing."
)


def ocr_image_png(png_bytes: bytes) -> str:
    """Read text from a page image using Gemini vision (raw SDK for reliability)."""
    config.require_api_key()
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(config.CHAT_MODEL)
    response = model.generate_content(
        [_OCR_INSTRUCTION, {"mime_type": "image/png", "data": png_bytes}],
        request_options={"timeout": config.REQUEST_TIMEOUT},
    )
    try:
        return (response.text or "").strip()
    except Exception:  # blocked/empty candidate
        return ""


# --------------------------------------------------------------------------
# PDF loading -> page-aware Documents (enables citations), with OCR fallback
# --------------------------------------------------------------------------
def load_and_split_pdf(
    file_path: str,
    enable_ocr: Optional[bool] = None,
    ocr_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Document], Dict[str, int]]:
    """Read a PDF into page-aware chunks. Returns (documents, stats).

    Native text is used when present; pages with little/no extractable text are
    rendered and OCR'd via Gemini vision (when enabled and a key is available).
    stats = {"pages": total_pages, "text_pages": n, "ocr_pages": n, "chunks": n}
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    enable_ocr = config.OCR_ENABLED if enable_ocr is None else enable_ocr
    can_ocr = enable_ocr and config.has_api_key()
    source = pdf_path.name

    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    page_texts: List[str] = []
    metadatas: List[dict] = []
    text_pages = ocr_pages = ocr_used = 0

    try:
        for index in range(total_pages):
            page = doc[index]
            text = (page.get_text() or "").strip()

            if len(text) < config.OCR_MIN_CHARS and can_ocr and ocr_used < config.OCR_MAX_PAGES:
                if ocr_progress:
                    ocr_progress(index + 1, total_pages)
                ocr_used += 1
                try:
                    pixmap = page.get_pixmap(dpi=config.OCR_DPI)
                    ocr_text = ocr_image_png(pixmap.tobytes("png"))
                    if len(ocr_text) > len(text):
                        text, _ocred = ocr_text, True
                        ocr_pages += 1
                except Exception as error:  # rate limit / transient -> skip page
                    logger.warning("OCR failed on %s p.%d: %s", source, index + 1, error)
            elif text:
                text_pages += 1

            if text:
                page_texts.append(text)
                metadatas.append({"source": source, "page": index + 1})
    finally:
        doc.close()

    if not page_texts:
        raise ValueError(
            f"No extractable text was found in '{source}'. "
            "If it is a scanned PDF, enable OCR (needs a valid API key)."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    documents = splitter.create_documents(page_texts, metadatas=metadatas)
    stats = {
        "pages": total_pages,
        "text_pages": text_pages,
        "ocr_pages": ocr_pages,
        "chunks": len(documents),
    }
    logger.info("Loaded '%s': %s", source, stats)
    return documents, stats


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------
def create_vector_store(documents: List[Document]) -> FAISS:
    if not documents:
        raise ValueError("No document chunks were provided to index.")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(config.INDEX_DIR))
    logger.info("Vector store created with %d chunks and saved to %s.",
                len(documents), config.INDEX_DIR)
    return vector_store


def load_vector_store() -> Optional[FAISS]:
    if not config.INDEX_DIR.exists():
        logger.warning("FAISS index not found at %s.", config.INDEX_DIR)
        return None
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        str(config.INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Vector store loaded from %s.", config.INDEX_DIR)
    return vector_store


# --------------------------------------------------------------------------
# Grounded Q&A (with citations + lightweight conversational memory)
# --------------------------------------------------------------------------
QA_PROMPT = (
    "You are a precise research assistant. Answer the QUESTION using ONLY the "
    "CONTEXT below, which was extracted from the user's PDF documents.\n\n"
    "Rules:\n"
    "- Use only facts stated in the context. Never use outside knowledge or "
    "guess.\n"
    "- If the answer is not contained in the context, reply exactly: "
    "\"The answer is not available in the provided document(s).\"\n"
    "- Be clear and thorough. Quote figures, names, dates and terms exactly.\n"
    "- Refer to page numbers from the context when they support your answer.\n\n"
    "CONTEXT:\n{context}\n\n"
    "EARLIER CONVERSATION (for reference only, may be empty):\n{history}\n\n"
    "QUESTION: {question}\n\nANSWER:"
)


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "document")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _format_history(chat_history: Optional[List[dict]], max_turns: int = 3) -> str:
    if not chat_history:
        return "(none)"
    recent = chat_history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
    return "\n\n".join(lines)


def _dedupe_sources(docs: List[Document]) -> List[Dict[str, object]]:
    seen = set()
    sources: List[Dict[str, object]] = []
    for doc in docs:
        key = (doc.metadata.get("source", "document"), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources


def retrieve(vector_store: FAISS, question: str, k: Optional[int] = None) -> List[Document]:
    if vector_store is None:
        raise ValueError("Vector store is not loaded. Process PDFs first.")
    return vector_store.similarity_search(question, k=k or config.RETRIEVAL_K)


def build_prompt_from_docs(
    question: str,
    docs: List[Document],
    chat_history: Optional[List[dict]] = None,
) -> str:
    """Build the grounded prompt from already-retrieved documents."""
    return QA_PROMPT.format(
        context=_format_context(docs),
        history=_format_history(chat_history),
        question=question,
    )


def build_answer_prompt(
    question: str,
    vector_store: FAISS,
    chat_history: Optional[List[dict]] = None,
    k: Optional[int] = None,
):
    """Retrieve context and build the final prompt. Returns (prompt, docs)."""
    docs = retrieve(vector_store, question, k=k)
    return build_prompt_from_docs(question, docs, chat_history), docs


def sources_from_docs(docs: List[Document]) -> List[Dict[str, object]]:
    """Public helper: de-duplicated (source, page) citations from documents."""
    return _dedupe_sources(docs)


def answer_question(
    question: str,
    vector_store: FAISS,
    chat_history: Optional[List[dict]] = None,
) -> Dict[str, object]:
    """Non-streaming answer. Returns {answer, sources, documents}."""
    prompt, docs = build_answer_prompt(question, vector_store, chat_history)
    llm = get_llm()
    response = llm.invoke(prompt)
    answer = (response.content or "").strip()
    logger.info("Answered question: %s...", question[:50])
    return {"answer": answer, "sources": _dedupe_sources(docs), "documents": docs}


def _chunk_to_text(content) -> str:
    """Normalize a message chunk's content to plain text.

    Gemini streaming sometimes yields content as a list of parts
    (e.g. [{"text": "..."}]) rather than a string; coerce both to text so
    st.write_stream never renders raw JSON.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def stream_answer(prompt: str) -> Iterator[str]:
    """Yield answer tokens for a prepared prompt, falling back to a single call."""
    llm = get_llm()
    try:
        for chunk in llm.stream(prompt):
            text = _chunk_to_text(getattr(chunk, "content", ""))
            if text:
                yield text
    except Exception as error:  # streaming unsupported/transient -> one-shot
        logger.warning("Streaming failed (%s); falling back to invoke.", error)
        yield _chunk_to_text(getattr(llm.invoke(prompt), "content", "")).strip()


# --------------------------------------------------------------------------
# Whole-document summarization (map -> reduce -> final), unlimited length
# --------------------------------------------------------------------------
_MAP_PROMPT = (
    "You are summarizing ONE excerpt from a larger document titled \"{title}\".\n"
    "Write a faithful, self-contained summary of the excerpt using ONLY "
    "information explicitly stated in it. Do NOT add facts, opinions, or "
    "outside knowledge. Preserve important names, numbers, dates and "
    "terminology exactly as written.\n\n"
    "EXCERPT (part {idx} of {total}):\n{text}\n\n"
    "FAITHFUL SUMMARY OF THIS EXCERPT:"
)

_REDUCE_PROMPT = (
    "You are compressing several section summaries of the document titled "
    "\"{title}\" into one shorter set of notes. Use ONLY the information in the "
    "section summaries below. Do NOT introduce new facts. Remove redundancy and "
    "keep key names, numbers and terminology.\n\n"
    "SECTION SUMMARIES:\n{text}\n\nCOMBINED NOTES:"
)

_FINAL_PROMPT = (
    "You are writing the final summary of the document titled \"{title}\".\n"
    "Using ONLY the information in the NOTES below, produce a well-structured "
    "summary. Do NOT add any outside information or invented details.\n\n"
    "Format exactly as:\n"
    "## Overview\n"
    "One short paragraph on the document's purpose and overall content.\n\n"
    "## Key Points\n"
    "- 5 to 12 concise bullet points of the most important information.\n\n"
    "## Details\n"
    "A few short paragraphs covering the main sections/themes in order.\n\n"
    "NOTES:\n{text}\n\nFINAL SUMMARY:"
)


def _total_chars(items: List[str]) -> int:
    return sum(len(x) for x in items)


def _group_by_chars(items: List[str], max_chars: int) -> List[List[str]]:
    """Greedily pack strings into groups no larger than max_chars."""
    groups: List[List[str]] = []
    current: List[str] = []
    size = 0
    for item in items:
        if current and size + len(item) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item)
    if current:
        groups.append(current)
    return groups


def _invoke_text(llm: ChatGoogleGenerativeAI, prompt: str, throttle: float) -> str:
    result = (llm.invoke(prompt).content or "").strip()
    if throttle:
        time.sleep(throttle)
    return result


def summarize_documents(
    documents: List[Document],
    title: str = "document",
    progress: Optional[ProgressFn] = None,
    throttle: Optional[float] = None,
) -> str:
    """Summarize any-length text via hierarchical map-reduce.

    Grounded (temperature ~0 + strict prompts) to minimize hallucination, and
    recursive so it handles book-length inputs without a context-window limit.
    """
    config.require_api_key()
    if not documents:
        raise ValueError("There is nothing to summarize.")

    throttle = config.SUMMARY_THROTTLE_SECONDS if throttle is None else throttle
    llm = get_llm(temperature=0.1)

    full_text = "\n\n".join(doc.page_content for doc in documents)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.SUMMARY_MAP_CHARS,
        chunk_overlap=config.SUMMARY_MAP_OVERLAP,
    )
    chunks = splitter.split_text(full_text)

    # MAP: summarize each excerpt independently.
    summaries: List[str] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        if progress:
            progress("map", idx, total)
        prompt = _MAP_PROMPT.format(title=title, idx=idx, total=total, text=chunk)
        summaries.append(_invoke_text(llm, prompt, throttle))

    # REDUCE: collapse in rounds until the notes comfortably fit one final call.
    while len(summaries) > 1 and _total_chars(summaries) > config.SUMMARY_GROUP_CHARS:
        groups = _group_by_chars(summaries, config.SUMMARY_GROUP_CHARS)
        if len(groups) >= len(summaries):  # safety: avoid non-shrinking loops
            break
        reduced: List[str] = []
        for gidx, group in enumerate(groups, start=1):
            if progress:
                progress("reduce", gidx, len(groups))
            prompt = _REDUCE_PROMPT.format(title=title, text="\n\n".join(group))
            reduced.append(_invoke_text(llm, prompt, throttle))
        summaries = reduced

    # FINAL: format the grounded notes into a structured summary.
    if progress:
        progress("final", 1, 1)
    notes = "\n\n".join(f"- {s}" for s in summaries)
    final_prompt = _FINAL_PROMPT.format(title=title, text=notes)
    return (llm.invoke(final_prompt).content or "").strip()
