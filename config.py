"""Central configuration for the PDF Research Assistant.

All tunables live here and can be overridden through environment variables
(loaded from the local .env file, or from the host/Streamlit secrets when
deployed). Import this module early so the API key and model names are
configured before any client is created.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load environment once, at import time.
load_dotenv(dotenv_path=ENV_PATH)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Persistent data locations (mount DATA_DIR as a volume when deployed) --
DATA_DIR = Path(_get_str("DATA_DIR", str(BASE_DIR / "data")))
INDEX_DIR = DATA_DIR / "faiss_index"
MANIFEST_PATH = DATA_DIR / "manifest.json"
UPLOAD_DIR = DATA_DIR / "uploaded_pdfs"

# --- Optional single-password gate (protects your API bill when deployed) -
# If unset/empty, the app is open (fine for local use).
APP_PASSWORD = _get_str("APP_PASSWORD", "")

# --- Models (verified available for this key on 2026-07-23) ---------------
# `gemini-1.5-pro` and `models/embedding-001` used by the original code are
# retired. These defaults are confirmed working; override via .env if needed.
CHAT_MODEL = _get_str("GEMINI_CHAT_MODEL", "gemini-flash-latest")
EMBEDDING_MODEL = _get_str("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
TEMPERATURE = _get_float("GEMINI_TEMPERATURE", 0.1)
REQUEST_TIMEOUT = _get_float("GEMINI_TIMEOUT", 120.0)
MAX_RETRIES = _get_int("GEMINI_MAX_RETRIES", 3)

# --- Retrieval / chunking for Q&A ----------------------------------------
CHUNK_SIZE = _get_int("CHUNK_SIZE", 1200)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 200)
RETRIEVAL_K = _get_int("RETRIEVAL_K", 5)

# --- OCR fallback for scanned / image-only PDF pages ---------------------
# Pages whose extractable text is shorter than OCR_MIN_CHARS are rendered to
# an image and read with Gemini vision. Set OCR_ENABLED=false to disable.
OCR_ENABLED = _get_bool("OCR_ENABLED", True)
OCR_MIN_CHARS = _get_int("OCR_MIN_CHARS", 20)
OCR_DPI = _get_int("OCR_DPI", 150)
OCR_MAX_PAGES = _get_int("OCR_MAX_PAGES", 300)  # safety cap on vision calls

# --- Summarization (scales to book-length documents) ---------------------
SUMMARY_MAP_CHARS = _get_int("SUMMARY_MAP_CHARS", 12000)
SUMMARY_MAP_OVERLAP = _get_int("SUMMARY_MAP_OVERLAP", 500)
SUMMARY_GROUP_CHARS = _get_int("SUMMARY_GROUP_CHARS", 9000)
# Optional pause between calls to stay under free-tier rate limits (seconds).
SUMMARY_THROTTLE_SECONDS = _get_float("SUMMARY_THROTTLE_SECONDS", 0.0)


def has_api_key() -> bool:
    """Return True if a Google API key is configured."""
    return bool(os.environ.get("GOOGLE_API_KEY"))


def require_api_key() -> None:
    """Raise a clear error if the Google API key is missing."""
    if not has_api_key():
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file "
            "(get a key at https://aistudio.google.com/apikey)."
        )
