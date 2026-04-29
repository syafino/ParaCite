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
from src.retrieve.cite import format_citation
from src.retrieve.search import SemanticSearch

log = logging.getLogger(__name__)


_MISSING_FEATURES = [
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
        style: str = "apa",
    ) -> dict[str, Any]:
        if not text or not text.strip():
            return {"claims": [], "missing": list(_MISSING_FEATURES), "style": style}

        claims = extract_claims(text)
        out: list[dict[str, Any]] = []
        for claim in claims:
            hits = self.search.query(claim["text"], top_k=top_k)
            sources = []
            for hit in hits:
                source = _shape_source(hit, claim.get("claim_id"))
                try:
                    formatted = format_citation(source, style)
                    source["citation"] = formatted["citation"]
                    source["warnings"] = formatted["warnings"]
                except Exception as e:
                    log.warning("Citation formatting failed: %s", e)
                    source["citation"] = ""
                    source["warnings"] = [str(e)]
                sources.append(source)

            out.append(
                {
                    **claim,
                    "sources": sources,
                }
            )

        return {
            "claims": out,
            "missing": list(_MISSING_FEATURES),
            "style": style,
        }


def _shape_source(hit: dict[str, Any], claim_id: str | None = None) -> dict[str, Any]:
    """Trim retrieval hit to a friendly response shape.

    The retriever returns ``{score, chunk_id, doc_id, text, metadata}``.
    We flatten metadata into the top level to match the formatter contract.
    """
    metadata = hit.get("metadata", {})
    text = hit.get("text", "") or ""

    # Start with metadata fields
    out = dict(metadata)

    # Override / add retrieval-specific fields
    out.update({
        "claim_id": claim_id or "",
        "score": hit.get("score"),
        "doc_id": hit.get("doc_id") or metadata.get("id"),
        "matched_chunk": text[:500],
    })

    return out
