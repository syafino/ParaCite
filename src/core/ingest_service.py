"""Synchronous ingest pipeline: file -> chunks -> embeddings -> FAISS.

The HTTP API runs ``IngestService.ingest`` inside a background thread (with
status updates flowing through ``JobRegistry``). The CLI calls it directly.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable

from src.embeddings.base import Embedder
from src.index.vector_store import VectorStore
from src.ingest.chunk_text import chunk_text
from src.ingest.extract_text import extract_text

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, dict[str, Any]], None]


class IngestService:
    """Pipeline: extract -> chunk -> embed -> add to VectorStore."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def ingest(
        self,
        path: Path,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Process ``path`` and append its chunks to the vector store.

        ``on_progress(stage, info)`` is called between pipeline stages so the
        async HTTP layer can update its job record. ``info`` is a dict with
        whatever fields are known at that stage (e.g. ``doc_id``,
        ``num_chunks``).
        """
        path = Path(path)

        def emit(stage: str, **info: Any) -> None:
            if on_progress is not None:
                on_progress(stage, info)

        emit("hashing")
        doc_id = _sha256_of_file(path)[:16]

        emit("extracting", doc_id=doc_id)
        text = extract_text(path)
        if not text.strip():
            raise ValueError(f"no extractable text in {path.name}")

        emit("chunking", doc_id=doc_id)
        chunks = chunk_text(doc_id, text, source_filename=path.name)
        if not chunks:
            raise ValueError(f"no chunks produced from {path.name}")

        emit("embedding", doc_id=doc_id, num_chunks=len(chunks))
        embeddings = self.embedder.embed([c["text"] for c in chunks])

        emit("indexing", doc_id=doc_id, num_chunks=len(chunks))
        self.store.add(
            chunks,
            embeddings,
            embedder_name=type(self.embedder).__name__,
        )

        emit("done", doc_id=doc_id, num_chunks=len(chunks))
        log.info("Ingested %s as doc_id=%s (%d chunks)", path.name, doc_id, len(chunks))
        return {"doc_id": doc_id, "num_chunks": len(chunks), "filename": path.name}


def _sha256_of_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
