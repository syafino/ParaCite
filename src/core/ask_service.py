"""Synchronous ask pipeline: text -> claims -> retrieved sources per claim.

Honest "wire only what exists" mode (per implementation plan): we run the
sentence parser to find citable claims and do a vector-only retrieval for
each. The downstream citation formatter and BM25 hybrid don't exist yet, so
we surface that explicitly via the ``missing`` field in the response.
"""

from __future__ import annotations

import logging
from typing import Any

from src.parse import extract_claims
from src.retrieve.search import SemanticSearch

log = logging.getLogger(__name__)


_MISSING_FEATURES = [
    "citation_formatter",
    "bm25_hybrid",
    "courtlistener_metadata_join",
]


class AskService:
    """Compose claim extraction + vector retrieval into a single response."""

    def __init__(self, search: SemanticSearch) -> None:
        self.search = search

    def ask(
        self,
        text: str,
        top_k: int = 3,
        style: str = "apa",  # accepted but unused until formatter lands
    ) -> dict[str, Any]:
        if not text or not text.strip():
            return {"claims": [], "missing": list(_MISSING_FEATURES), "style": style}

        claims = extract_claims(text)
        out: list[dict[str, Any]] = []
        for claim in claims:
            hits = self.search.query(claim["text"], top_k=top_k)
            out.append(
                {
                    **claim,
                    "sources": [_shape_source(h) for h in hits],
                }
            )

        return {
            "claims": out,
            "missing": list(_MISSING_FEATURES),
            "style": style,
        }


def _shape_source(hit: dict[str, Any]) -> dict[str, Any]:
    """Trim retrieval hit to a friendly response shape.

    The retriever returns ``{score, chunk_id, doc_id, text, metadata}``; we
    rename ``text`` -> ``matched_chunk`` (capped) so the response matches the
    plan's documented contract and stays compact over the wire.
    """
    text = hit.get("text", "") or ""
    return {
        "score": hit.get("score"),
        "chunk_id": hit.get("chunk_id"),
        "doc_id": hit.get("doc_id"),
        "matched_chunk": text[:500],
        "metadata": hit.get("metadata", {}),
    }
