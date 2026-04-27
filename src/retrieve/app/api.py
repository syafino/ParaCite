"""Adapter boundary between the Streamlit frontend and backend pipeline.

The parser/retriever/formatter modules are still being built by separate
workstreams, so this module is intentionally defensive: it tries real hooks
when they exist and otherwise returns deterministic demo results that keep the
frontend usable.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from typing import Any, Callable

SUPPORTED_STYLES = {"bluebook", "apa", "mla", "ieee", "bibtex"}
DEFAULT_STYLE = "bluebook"


def run_citation_pipeline(text: str, style: str, top_k: int = 3) -> dict:
    """Run the citation pipeline for pasted text.

    Returns the stable frontend contract described in ``implementation.md``.
    When backend pieces are unavailable, the response status is ``"demo"`` and
    deterministic citation placeholders are returned.
    """
    cleaned_text = text.strip()
    normalized_style = _normalize_style(style)
    safe_top_k = max(1, min(int(top_k or 1), 10))

    if not cleaned_text:
        return {
            "status": "error",
            "message": "Paste text before asking ParaCite for citations.",
            "claims": [],
        }

    try:
        claims, parser_message = _get_claims(cleaned_text)
        if not claims:
            return {
                "status": "demo",
                "message": "No claim spans were detected, so demo citations are hidden.",
                "claims": [],
            }

        backend_sources = _try_retrieve_sources(claims, safe_top_k)
        if backend_sources is None:
            return _demo_response(claims, normalized_style, safe_top_k, parser_message)

        formatted_claims = _format_real_sources(
            claims=claims,
            sources_by_claim=backend_sources,
            style=normalized_style,
        )
        return {
            "status": "ok",
            "message": "Citations generated from the available backend modules.",
            "claims": formatted_claims,
        }
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        return {
            "status": "error",
            "message": f"Could not run the citation pipeline: {exc}",
            "claims": [],
        }


def _normalize_style(style: str) -> str:
    candidate = (style or DEFAULT_STYLE).strip().lower()
    return candidate if candidate in SUPPORTED_STYLES else DEFAULT_STYLE


def _get_claims(text: str) -> tuple[list[dict], str]:
    parser = _find_callable(
        [
            ("src.parse.claims", "extract_claims"),
            ("src.parse.claims", "parse_claims"),
            ("src.retrieve.claims", "extract_claims"),
            ("src.retrieve.claims", "parse_claims"),
        ]
    )
    if parser is None:
        return _fallback_claims(text), "Using frontend demo claim detection."

    claims = parser(text)
    return [_normalize_claim(claim, idx, text) for idx, claim in enumerate(claims)], (
        "Using parser module output."
    )


def _try_retrieve_sources(claims: list[dict], top_k: int) -> dict[str, list[dict]] | None:
    retriever = _find_callable(
        [
            ("src.retrieve.hybrid", "retrieve_for_claims"),
            ("src.retrieve.hybrid", "retrieve_claims"),
            ("src.retrieve.api", "retrieve_for_claims"),
        ]
    )
    if retriever is None:
        return None

    raw_results = retriever(claims, top_k=top_k)
    if isinstance(raw_results, dict):
        return {
            str(claim_id): list(sources or [])
            for claim_id, sources in raw_results.items()
        }

    grouped: dict[str, list[dict]] = {claim["claim_id"]: [] for claim in claims}
    for source in raw_results or []:
        claim_id = str(source.get("claim_id", ""))
        if claim_id in grouped:
            grouped[claim_id].append(source)
    return grouped


def _format_real_sources(
    claims: list[dict],
    sources_by_claim: dict[str, list[dict]],
    style: str,
) -> list[dict]:
    formatter = _find_formatter(style)

    formatted_claims: list[dict] = []
    for claim in claims:
        citations = []
        for source in sources_by_claim.get(claim["claim_id"], []):
            if formatter is None:
                citation = _citation_from_source(source, style)
            else:
                citation = formatter(source, style)
            citations.append(_normalize_citation(citation, source, style))
        formatted_claims.append({**claim, "citations": citations})
    return formatted_claims


def _fallback_claims(text: str) -> list[dict]:
    matches = list(re.finditer(r"[^.!?\n]+(?:[.!?]|$)", text))
    if not matches:
        return [_normalize_claim({"text": text, "char_start": 0, "char_end": len(text)}, 0, text)]

    claims: list[dict] = []
    for idx, match in enumerate(matches[:5]):
        claim_text = match.group(0).strip()
        if not claim_text:
            continue
        char_start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        char_end = char_start + len(claim_text)
        claims.append(
            {
                "claim_id": f"c{idx}",
                "text": claim_text,
                "char_start": char_start,
                "char_end": char_end,
                "context": text[max(0, char_start - 120): min(len(text), char_end + 120)],
            }
        )
    return claims


def _demo_response(claims: list[dict], style: str, top_k: int, parser_message: str) -> dict:
    demo_claims = []
    for claim in claims:
        demo_claims.append(
            {
                **claim,
                "citations": [
                    _demo_citation(claim=claim, style=style, rank=rank)
                    for rank in range(1, top_k + 1)
                ],
            }
        )

    return {
        "status": "demo",
        "message": (
            f"{parser_message} Retriever/formatter modules are not fully available yet, "
            "so these are deterministic demo citations."
        ),
        "claims": demo_claims,
    }


def _demo_citation(claim: dict, style: str, rank: int) -> dict:
    digest = hashlib.sha1(f"{claim['claim_id']}:{claim['text']}:{rank}".encode("utf-8")).hexdigest()
    doc_id = f"demo-{digest[:8]}"
    case_name = _demo_case_name(rank)
    reporter = f"{300 + rank} U.S. {480 + rank}"
    year = str(1950 + rank)
    court_id = "scotus"
    url = f"https://www.courtlistener.com/opinion/{doc_id}/"

    source = {
        "claim_id": claim["claim_id"],
        "doc_id": doc_id,
        "style": style,
        "case_name": case_name,
        "case_name_short": case_name,
        "date_filed": f"{year}-01-01",
        "court_id": court_id,
        "citations": [reporter],
        "cluster_url": url,
        "score": round(max(0.1, 0.92 - ((rank - 1) * 0.11)), 2),
        "matched_chunk": claim["text"][:180],
    }
    return {
        "doc_id": doc_id,
        "style": style,
        "citation": _citation_from_source(source, style),
        "warnings": ["demo citation; backend retrieval/formatting not connected"],
        "score": source["score"],
        "matched_chunk": source["matched_chunk"],
        "cluster_url": url,
    }


def _demo_case_name(rank: int) -> str:
    names = [
        "Brown v. Board of Education",
        "Miranda v. Arizona",
        "Gideon v. Wainwright",
        "Mapp v. Ohio",
        "Katz v. United States",
    ]
    return names[(rank - 1) % len(names)]


def _citation_from_source(source: dict, style: str) -> str:
    case_name = source.get("case_name_short") or source.get("case_name") or "Unknown Case"
    citations = source.get("citations") or []
    reporter = citations[0] if citations else "No reporter citation"
    year = str(source.get("date_filed", ""))[:4] or "n.d."
    court_id = source.get("court_id") or "court"
    url = source.get("cluster_url") or ""

    if style == "apa":
        return f"{case_name} ({year}). {reporter}. {url}".strip()
    if style == "mla":
        date = source.get("date_filed") or year
        return f"{case_name}. {court_id}, {date}, {url}.".strip()
    if style == "ieee":
        return f"{case_name}, {reporter}, {court_id}, {year}. {url}".strip()
    if style == "bibtex":
        key = re.sub(r"[^A-Za-z0-9]+", "", str(source.get("doc_id") or "paracite"))
        return (
            f"@misc{{{key},\n"
            f"  title = {{{case_name}}},\n"
            f"  year = {{{year}}},\n"
            f"  url = {{{url}}},\n"
            f"  note = {{{reporter}}}\n"
            "}"
        )
    return f"{case_name}, {reporter} ({court_id} {year})."


def _normalize_claim(claim: dict, idx: int, full_text: str) -> dict:
    claim_text = str(claim.get("text", "")).strip()
    char_start = int(claim.get("char_start", full_text.find(claim_text) if claim_text else 0))
    if char_start < 0:
        char_start = 0
    char_end = int(claim.get("char_end", char_start + len(claim_text)))
    return {
        "claim_id": str(claim.get("claim_id") or f"c{idx}"),
        "text": claim_text,
        "char_start": char_start,
        "char_end": char_end,
        "context": claim.get("context"),
    }


def _normalize_citation(citation: dict, source: dict, style: str) -> dict:
    if isinstance(citation, str):
        citation = {
            "doc_id": source.get("doc_id", ""),
            "style": style,
            "citation": citation,
            "warnings": [],
        }
    elif not isinstance(citation, dict):
        citation = {
            "doc_id": source.get("doc_id", ""),
            "style": style,
            "citation": _citation_from_source(source, style),
            "warnings": ["formatter returned an unsupported result; used frontend fallback"],
        }

    return {
        "doc_id": str(citation.get("doc_id") or source.get("doc_id") or ""),
        "style": str(citation.get("style") or style),
        "citation": str(citation.get("citation") or _citation_from_source(source, style)),
        "warnings": list(citation.get("warnings") or []),
        "score": source.get("score"),
        "matched_chunk": source.get("matched_chunk") or source.get("text") or "",
        "cluster_url": source.get("cluster_url") or "",
    }


def _find_callable(candidates: list[tuple[str, str]]) -> Callable[..., Any] | None:
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        func = getattr(module, attr_name, None)
        if callable(func):
            return func
    return None


def _find_formatter(style: str) -> Callable[..., Any] | None:
    style_module = "bibtex" if style == "bibtex" else f"format_{style}"
    style_func = "format_bibtex" if style == "bibtex" else f"format_{style}"
    return _find_callable(
        [
            ("src.retrieve.cite", "format_citation"),
            ("cite", "format_citation"),
            (f"src.retrieve.cite.{style_module}", "format_citation"),
            (f"src.retrieve.cite.{style_module}", style_func),
            (f"cite.{style_module}", "format_citation"),
            (f"cite.{style_module}", style_func),
        ]
    )
