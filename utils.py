"""Small helpers shared across the app."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> bool:
    """Load variables from the local .env file (idempotent)."""
    load_dotenv(dotenv_path=Path(".env"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return True


def human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
