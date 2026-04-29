# MLA 9th edition for legal cases.
# "<Case Name>. <Court long name>, <date>, <url>."

from . import courts
from . import dates


def _name(src):
    return (src.get("case_name")
            or src.get("case_name_full")
            or src.get("case_name_short")
            or "").strip()


def render(src):
    warnings = []

    name = _name(src)
    if not name:
        warnings.append("no case name")

    court = courts.long_name(src.get("court_id", ""))
    pd = dates.parse(src.get("date_filed", ""))
    when = dates.mla_long(pd)
    if not when:
        warnings.append("no date")

    url = (src.get("cluster_url") or "").strip()

    pieces = []
    if name:
        pieces.append(f"{name}.")
    middle = ", ".join(x for x in (court, when) if x)
    if middle:
        pieces.append(f"{middle},")
    if url:
        pieces.append(f"{url}.")

    text = " ".join(pieces).strip().rstrip(",")

    return {
        "claim_id": src.get("claim_id", ""),
        "doc_id":   str(src.get("doc_id", "")),
        "style":    "mla",
        "citation": text,
        "warnings": warnings,
    }
