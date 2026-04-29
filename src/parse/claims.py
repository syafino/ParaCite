"""Top-level orchestrator: text -> list of Claim dicts (per §3.1).

Pipeline:
    user text -> split_sentences -> classify_batch -> assemble Claim dicts

Returned Claim shape (matches implementation.md §3.1 exactly)::

    {
        "claim_id": "c0",
        "text": "<sentence>",
        "char_start": 12,
        "char_end": 47,
        "context": "<prev sent> <next sent>"  # or None
    }
"""

from __future__ import annotations

from typing import Any

from src.parse.llm import classify_batch
from src.parse.splitter import split_sentences


def _build_context(
    sentences: list[tuple[str, int, int]], idx: int
) -> str | None:
    parts: list[str] = []
    if idx - 1 >= 0:
        parts.append(sentences[idx - 1][0])
    if idx + 1 < len(sentences):
        parts.append(sentences[idx + 1][0])
    if not parts:
        return None
    ctx = " ".join(p.strip() for p in parts if p.strip())
    return ctx or None


def extract_claims(text: str) -> list[dict[str, Any]]:
    """Return the list of citable Claims found in ``text``.

    Each claim follows the §3.1 contract: ``claim_id``, ``text``,
    ``char_start``, ``char_end``, ``context``.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    judgements = classify_batch([s for s, _, _ in sentences])

    claims: list[dict[str, Any]] = []
    next_id = 0
    for idx, ((sent, start, end), j) in enumerate(zip(sentences, judgements)):
        if not j.get("citable"):
            continue
        claims.append(
            {
                "claim_id": f"c{next_id}",
                "text": sent,
                "char_start": start,
                "char_end": end,
                "context": _build_context(sentences, idx),
            }
        )
        next_id += 1

    return claims


if __name__ == "__main__":
    import json

    sample = (
        "Transformers have become the dominant architecture in NLP. "
        "In this paper, we propose a new method for citation extraction. "
        "We trained for 10 epochs on a single GPU. "
        "BERT achieved state-of-the-art results on the GLUE benchmark in 2018."
    )
    result = extract_claims(sample)
    print(json.dumps(result, indent=2))
