# 📚 PDF Research Assistant

A RAG-based PDF assistant built with **Streamlit**, **LangChain**, **Google Gemini**, **FAISS**, and **PyMuPDF**.

Build a personal library of PDFs, then:

- **💬 Ask** questions answered *only* from your documents, with **page-level citations**, **conversational memory**, and the ability to **scope** the search to specific documents.
- **📝 Summarize** an entire document — even a full book — with a **hallucination-resistant, recursive map-reduce** summary that has **no length limit**.

## Features

- **Persistent document library** — add, list, scope, and remove PDFs individually. The index and manifest are saved to disk, so your library survives restarts and redeploys.
- **OCR for scanned PDFs** — image-only pages are rendered with PyMuPDF and read with Gemini vision, so scanned documents work too (no system dependencies required).
- **Grounded answers with citations** — every answer is constrained to retrieved context and cites the source file + page. If the answer isn't in your PDFs, it says so instead of guessing.
- **Whole-document summaries at any length** — long documents are summarized map → reduce → final and recursively collapsed, so a 500-page book works like a 2-page memo.
- **Streaming answers** and progress indicators for long summaries.
- **Optional password gate** — set `APP_PASSWORD` to protect your API bill when deployed publicly.
- **Deploy-ready** — Dockerfile, `.streamlit/config.toml`, and Streamlit Cloud secrets support included.
- **Safer** — dynamic content renders as plain markdown (no HTML injection); secrets and PDFs are kept out of version control.

## Quick start (local)

```bash
cd ~/pdf_rag_assistant
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
cp .env.example .env          # then edit .env and set GOOGLE_API_KEY
streamlit run app.py
```

Get a key at [Google AI Studio](https://aistudio.google.com/apikey) and paste it into `.env` as `GOOGLE_API_KEY`.

## Usage

1. In the sidebar, upload PDFs and click **➕ Add to library** (toggle **OCR scanned pages** for image-only PDFs).
2. **Ask tab** — pick which documents to search (default: all), type a question; the answer streams in with cited pages.
3. **Summarize tab** — choose a document (or *All documents*) and click **✨ Generate summary**; download it as Markdown.
4. Remove a document any time with the 🗑️ button next to it in the Library.

## Deployment

### Option A — Docker (self-host anywhere)

```bash
docker build -t pdf-assistant .
docker run -p 8501:8501 --env-file .env -v pdf_data:/app/data pdf-assistant
```

The named volume `pdf_data` persists your library across restarts. Open <http://localhost:8501>.

### Option B — Streamlit Community Cloud (free, fastest)

1. Push the repo to GitHub (the `.gitignore` keeps `.env`, PDFs, and the index out).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py`.
3. In **Advanced settings → Secrets**, paste (see `.streamlit/secrets.toml.example`):
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   APP_PASSWORD  = "choose-a-strong-password"
   ```
4. Deploy. The app reads secrets automatically.

> **Note:** Streamlit Cloud storage is ephemeral — the library resets when the app sleeps/redeploys. For durable storage, use Docker with a mounted volume (or point `DATA_DIR` at persistent storage).

## Configuration reference

All optional except the key; set in `.env` or secrets:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** Gemini API key. |
| `APP_PASSWORD` | *(empty)* | If set, gates the app behind a password. |
| `GEMINI_CHAT_MODEL` | `gemini-flash-latest` | Chat/generation + OCR model. |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model. |
| `OCR_ENABLED` | `true` | OCR fallback for scanned pages. |
| `OCR_DPI` / `OCR_MAX_PAGES` | `150` / `300` | OCR render quality / page cap. |
| `RETRIEVAL_K` | `5` | Chunks retrieved per question. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Q&A chunking. |
| `SUMMARY_MAP_CHARS` | `12000` | Excerpt size for the summary map step. |
| `SUMMARY_THROTTLE_SECONDS` | `0` | Pause between calls to respect rate limits. |
| `DATA_DIR` | `data` | Where the library + index are stored. |

## Evaluation

Answer quality is capped by *retrieval* quality — if the right page never enters
the context, the model can't cite it. `evaluation.py` measures retrieval
directly, at the same `(source, page)` citation granularity the UI shows, against
a small labelled gold set (`evaluation_gold_example.json`):

```bash
python evaluation.py --gold evaluation_gold_example.json --k 5
```

```text
Retrieval Evaluation  (k=5, 3 questions)
  Recall@5      0.500
  Precision@5   0.333   (citation accuracy)
  Hit-rate@5    0.667
  MRR           0.667
  MAP           0.500
```

Metrics are standard IR measures (recall@k, precision@k, hit-rate, MRR, MAP)
computed per question then averaged. The retriever is injected as a callable, so
the metrics are unit-tested with a stub — no index or API key required (see
`tests/test_evaluation.py`). Use it to tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, and
`RETRIEVAL_K` against a number instead of by feel.

## Notes

- Uses `faiss-cpu` (works on macOS Apple Silicon). PyMuPDF ships wheels, so OCR needs no system packages.
- **Free-tier rate limits:** summarizing a long book makes many calls and may hit `429` quota errors. Set `SUMMARY_THROTTLE_SECONDS` (e.g. `2`) to pace requests, or use a paid tier.
- The library (`data/`) and any uploaded PDFs are git-ignored — they can contain private information.
