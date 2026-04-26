"""Sentence-window chunker for uploaded documents.

Uses the same Punkt tokenizer as ``src.parse.splitter`` to split the document
into sentences, then greedily packs them into chunks of roughly
``target_chars`` characters. Each emitted chunk matches the schema expected by
``src.index.vector_store`` (``chunk_id``, ``doc_id``, ``text``, ``metadata``).
"""

from __future__ import annotations

from typing import Any

from src.parse.splitter import split_sentences


def chunk_text(
    doc_id: str,
    text: str,
    target_chars: int = 500,
    source_filename: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Pack sentences into ~``target_chars`` chunks.

    Returns a list of chunk dicts ready to write to ``chunks.jsonl`` and feed
    into ``VectorStore.add``.
    """
    metadata_base: dict[str, Any] = {"source": "upload"}
    if source_filename:
        metadata_base["filename"] = source_filename
    if extra_metadata:
        metadata_base.update(extra_metadata)

    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    buffer_len = 0
    chunk_idx = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, chunk_idx
        if not buffer:
            return
        chunk_text_value = " ".join(s.strip() for s in buffer if s.strip())
        if chunk_text_value:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}-c{chunk_idx}",
                    "doc_id": doc_id,
                    "text": chunk_text_value,
                    "metadata": dict(metadata_base),
                }
            )
            chunk_idx += 1
        buffer = []
        buffer_len = 0

    for sent, _start, _end in sentences:
        s_len = len(sent)
        if buffer and buffer_len + s_len > target_chars:
            flush()
        buffer.append(sent)
        buffer_len += s_len
    flush()

    return chunks
