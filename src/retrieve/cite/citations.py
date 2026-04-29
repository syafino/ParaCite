# Reporter citation parsing.
#
# Heads up: collect_metadata.py over in ingest does `str(c)` on dict-shaped
# citations when there's no top-level "cite" key. So instead of a clean
# "347 U.S. 483" we sometimes get the repr of the dict back as a string.
# This module recovers from that. Should probably be fixed upstream eventually.

import ast


def normalize(raw):
    """Return a clean reporter cite ('volume reporter page') or None."""
    if isinstance(raw, dict):
        d = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        # already a clean string?
        if not s.startswith("{"):
            return s
        try:
            d = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(d, dict):
            return None
    else:
        return None

    # if the dict has a 'cite' field, trust it
    if d.get("cite"):
        return str(d["cite"]).strip()

    vol = str(d.get("volume", "")).strip()
    rep = str(d.get("reporter", "")).strip()
    pg  = str(d.get("page", "")).strip()
    parts = [x for x in (vol, rep, pg) if x]
    if not parts:
        return None
    return " ".join(parts)


def first_normalized(citations):
    # walk the list and return the first one we can actually parse
    for c in citations or []:
        cleaned = normalize(c)
        if cleaned:
            return cleaned
    return None
