"""Sentence splitting with character offsets back into the original input.

Uses nltk's Punkt tokenizer via ``span_tokenize`` so we can preserve exact
``char_start`` / ``char_end`` offsets as required by the Claim contract
(section 3.1 of implementation.md).
"""

from __future__ import annotations

import nltk
from nltk.tokenize import PunktSentenceTokenizer

_TOKENIZER: PunktSentenceTokenizer | None = None


def _ensure_punkt() -> None:
    """Ensure the punkt tokenizer data is available; download lazily if not."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def _get_tokenizer() -> PunktSentenceTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _ensure_punkt()
        _TOKENIZER = PunktSentenceTokenizer()
    return _TOKENIZER


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into sentences with character offsets.

    Returns a list of ``(sentence, char_start, char_end)`` tuples where the
    offsets index into the original ``text`` (``text[char_start:char_end]``
    reproduces the sentence exactly).
    """
    if not text or not text.strip():
        return []

    tokenizer = _get_tokenizer()
    spans = tokenizer.span_tokenize(text)
    out: list[tuple[str, int, int]] = []
    for start, end in spans:
        sent = text[start:end]
        if sent.strip():
            out.append((sent, start, end))
    return out
