"""Persistent, multi-document FAISS store with per-document management.

Wraps a single FAISS index plus a JSON manifest so documents can be added,
listed, scoped, and removed individually — and survive restarts/redeploys when
DATA_DIR is a mounted volume.
"""

import json
import shutil
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS

import config
from rag_pipeline import get_embeddings, load_and_split_pdf

logger = logging.getLogger(__name__)


class DocumentStore:
    def __init__(self, vector_store: Optional[FAISS] = None,
                 manifest: Optional[Dict[str, dict]] = None):
        self.vector_store = vector_store
        self.manifest: Dict[str, dict] = manifest or {}

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls) -> "DocumentStore":
        """Load a previously persisted store, or return an empty one."""
        if config.INDEX_DIR.exists() and config.MANIFEST_PATH.exists():
            try:
                vector_store = FAISS.load_local(
                    str(config.INDEX_DIR), get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                manifest = json.loads(config.MANIFEST_PATH.read_text())
                logger.info("Loaded store with %d document(s).", len(manifest))
                return cls(vector_store, manifest)
            except Exception as error:
                logger.warning("Could not load existing store: %s", error)
        return cls()

    def persist(self) -> None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        if self.vector_store is not None and self.manifest:
            self.vector_store.save_local(str(config.INDEX_DIR))
            config.MANIFEST_PATH.write_text(json.dumps(self.manifest, indent=2))
        else:  # empty -> clean up any stale artifacts
            if config.INDEX_DIR.exists():
                shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
            if config.MANIFEST_PATH.exists():
                config.MANIFEST_PATH.unlink()

    # -- queries about contents -------------------------------------------
    def is_empty(self) -> bool:
        return not self.manifest or self.vector_store is None

    def names(self) -> List[str]:
        return list(self.manifest.keys())

    def list_documents(self) -> List[dict]:
        return [{"name": name, **info} for name, info in self.manifest.items()]

    def has_document(self, name: str) -> bool:
        return name in self.manifest

    def _docstore_items(self):
        # InMemoryDocstore keeps {id: Document} in _dict.
        return self.vector_store.docstore._dict.items() if self.vector_store else []

    def documents_for(self, name: str) -> List[Document]:
        docs = [d for _id, d in self._docstore_items()
                if d.metadata.get("source") == name]
        docs.sort(key=lambda d: d.metadata.get("page", 0))
        return docs

    def all_documents(self) -> List[Document]:
        docs = [d for _id, d in self._docstore_items()]
        docs.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("page", 0)))
        return docs

    # -- mutations ---------------------------------------------------------
    def add_pdf(self, file_path: str, display_name: str,
                enable_ocr: Optional[bool] = None,
                ocr_progress: Optional[Callable[[int, int], None]] = None) -> dict:
        """Index a PDF and merge it into the store. Replaces a same-named doc."""
        documents, stats = load_and_split_pdf(
            file_path, enable_ocr=enable_ocr, ocr_progress=ocr_progress
        )
        for doc in documents:
            doc.metadata["source"] = display_name  # normalize to the display name

        if self.has_document(display_name):
            self.remove_document(display_name, persist=False)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, get_embeddings())
        else:
            self.vector_store.add_documents(documents)

        info = {**stats, "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.manifest[display_name] = info
        self.persist()
        return info

    def remove_document(self, name: str, persist: bool = True) -> None:
        if self.vector_store is not None:
            ids = [_id for _id, d in self._docstore_items()
                   if d.metadata.get("source") == name]
            if ids:
                self.vector_store.delete(ids)
        self.manifest.pop(name, None)
        if not self.manifest:
            self.vector_store = None
        if persist:
            self.persist()

    # -- retrieval ---------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None,
               sources: Optional[List[str]] = None) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("No documents are indexed yet.")
        k = k or config.RETRIEVAL_K
        selected = set(sources) if sources else None
        # No scoping (or all docs selected): plain search.
        if not selected or selected >= set(self.manifest.keys()):
            return self.vector_store.similarity_search(query, k=k)
        # Scope to a subset via a metadata filter, over-fetching to fill k.
        return self.vector_store.similarity_search(
            query, k=k,
            filter=lambda meta: meta.get("source") in selected,
            fetch_k=max(k * 8, 40),
        )
