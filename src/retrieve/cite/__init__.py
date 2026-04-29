# Public entry point for the cite package.
# Usage:
#     from src.retrieve.cite import format_citation
#     formatted = format_citation(retrieved_source, "bluebook")

from .bibtex import render as _bibtex
from .format_apa import render as _apa
from .format_bluebook import render as _bluebook
from .format_mla import render as _mla


_FORMATTERS = {
    "apa":      _apa,
    "bibtex":   _bibtex,
    "bluebook": _bluebook,
    "mla":      _mla,
}

AVAILABLE_STYLES = sorted(_FORMATTERS)


def format_citation(source, style):
    key = (style or "").lower().strip()
    fn = _FORMATTERS.get(key)
    if fn is None:
        raise ValueError(f"Unknown style: {style!r}. Available: {AVAILABLE_STYLES}")
    return fn(source)


__all__ = ["format_citation", "AVAILABLE_STYLES"]
