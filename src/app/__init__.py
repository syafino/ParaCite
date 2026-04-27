"""Process-wide service singletons shared by the HTTP API and CLI.

Both ``src.app.api`` and ``src.app.cli`` get their ``IngestService`` /
``AskService`` through the accessors here. The embedder and FAISS index are
expensive to load (~1-3 seconds for sentence-transformers, plus FAISS file
IO), so we want exactly one instance per process regardless of how many
requests or commands flow through.
"""

from __future__ import annotations

import logging
import threading

from src.config import INDEX_DIR
from src.core import AskService, IngestService, JobRegistry
from src.embeddings.base import Embedder
from src.embeddings.sentence_transformer import SentenceTransformerEmbedder
from src.index.vector_store import VECTORS_DIR, VectorStore
from src.retrieve.search import SemanticSearch

log = logging.getLogger(__name__)

_INIT_LOCK = threading.RLock()
_EMBEDDER: Embedder | None = None
_STORE: VectorStore | None = None
_SEARCH: SemanticSearch | None = None
_INGEST: IngestService | None = None
_ASK: AskService | None = None
_JOBS: JobRegistry | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        with _INIT_LOCK:
            if _EMBEDDER is None:
                log.info("Loading SentenceTransformer embedder")
                _EMBEDDER = SentenceTransformerEmbedder()
    return _EMBEDDER


def get_store() -> VectorStore:
    """Return the shared VectorStore, creating an empty one if no index exists."""
    global _STORE
    if _STORE is None:
        with _INIT_LOCK:
            if _STORE is None:
                INDEX_DIR.mkdir(parents=True, exist_ok=True)
                VECTORS_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    log.info("Loading existing vector index from %s", VECTORS_DIR)
                    _STORE = VectorStore.load(VECTORS_DIR)
                except FileNotFoundError:
                    embedder = get_embedder()
                    log.info(
                        "No vector index found; creating empty one at %s (dim=%d)",
                        VECTORS_DIR, embedder.dim,
                    )
                    _STORE = VectorStore.empty(embedder.dim, output_dir=VECTORS_DIR)
    return _STORE


def get_ingest_service() -> IngestService:
    global _INGEST
    if _INGEST is None:
        with _INIT_LOCK:
            if _INGEST is None:
                _INGEST = IngestService(get_embedder(), get_store())
    return _INGEST


def get_ask_service() -> AskService:
    global _ASK, _SEARCH
    if _ASK is None:
        with _INIT_LOCK:
            if _ASK is None:
                _SEARCH = SemanticSearch(get_embedder(), get_store())
                _ASK = AskService(_SEARCH)
    return _ASK


def get_jobs() -> JobRegistry:
    global _JOBS
    if _JOBS is None:
        with _INIT_LOCK:
            if _JOBS is None:
                _JOBS = JobRegistry()
    return _JOBS


__all__ = [
    "get_ask_service",
    "get_embedder",
    "get_ingest_service",
    "get_jobs",
    "get_store",
]
