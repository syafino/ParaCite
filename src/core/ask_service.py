"""Synchronous ask pipeline: text -> claims -> hybrid retrieved sources."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import METADATA_DIR
from src.parse import extract_claims
from src.retrieve.cite import format_citation
from src.retrieve.hybrid import HybridSearch

log = logging.getLogger(__name__)


_MISSING_FEATURES = []


class AskService:
    """Compose claim extraction + hybrid retrieval into a single response."""

    def __init__(self, search: HybridSearch) -> None:
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

                doc_id = source.get("doc_id")
                if doc_id:
                    meta_path = METADATA_DIR / f"cluster_{doc_id}_meta.json"
                    if meta_path.exists():
                        try:
                            with meta_path.open("r", encoding="utf-8") as f:
                                meta_data = json.load(f)
                                for k, v in meta_data.items():
                                    if k not in source:
                                        source[k] = v
                        except Exception as e:
                            log.warning("Failed to load metadata for doc_id %s: %s", doc_id, e)
                
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
                    "citations": sources,
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
    metadata = _flatten_metadata(hit.get("metadata", {}))
    text = hit.get("text", "") or ""

    out = dict(metadata)

    out.update({
        "claim_id": claim_id or "",
        "score": hit.get("score"),
        "scores": hit.get("scores") or {},
        "retrieval_modes": hit.get("retrieval_modes") or [],
        "chunk_id": hit.get("chunk_id") or "",
        "doc_id": hit.get("doc_id") or metadata.get("id"),
        "matched_chunk": text[:500],
    })

    return out


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Handle both flat metadata and records shaped as {"metadata": {...}}."""
    if not isinstance(metadata, dict):
        return {}
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        merged = dict(nested)
        for key, value in metadata.items():
            if key != "metadata" and key not in merged:
                merged[key] = value
        return merged
    return dict(metadata)
