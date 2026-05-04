"""Load JSON document corpora into the ingest pipeline.

Supported shapes:

* a single object with a ``text`` field
* a list of objects with ``text`` fields
* an object containing a list under ``papers``, ``documents``, or ``records``

Each object can include metadata such as ``doc_id``, ``title``, ``authors``,
``year``, and ``url``. The loader keeps that metadata so retrieval results can
be cited and inspected by the frontend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_documents(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _extract_rows(payload)

    documents: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        text = str(row.get("text") or row.get("body") or row.get("content") or "").strip()
        if not text:
            continue

        doc_id = str(
            row.get("doc_id")
            or row.get("paper_id")
            or row.get("id")
            or f"{path.stem}_{idx:04d}"
        )
        title = str(row.get("title") or "").strip()
        year = row.get("year")
        url = str(row.get("url") or row.get("source_url") or "").strip()
        authors = row.get("authors") or row.get("author") or []
        if isinstance(authors, str):
            authors = [authors]
        elif not isinstance(authors, list):
            authors = []

        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"text", "body", "content", "chunk_id", "doc_id", "paper_id", "id"}
        }
        metadata.update(
            {
                "source": "json",
                "filename": path.name,
                "title": title,
                "authors": authors,
                "year": year,
                "url": url,
            }
        )

        if title:
            metadata.setdefault("case_name", title)
            metadata.setdefault("case_name_short", title)
        if year:
            metadata.setdefault("date_filed", str(year))
        if url:
            metadata.setdefault("cluster_url", url)

        documents.append({"doc_id": doc_id, "text": text, "metadata": metadata})

    return documents


def _extract_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("papers", "documents", "records"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
        if "text" in payload:
            return [payload]
    raise ValueError("JSON ingest requires a document object or a list of document objects with text fields")
