# APA 7th edition for legal cases.
# Format roughly: "<Case Name> (<year>). <reporter>. <url>"

from . import citations as cit
from . import dates


def _name(src):
    return (
        src.get("case_name")
        or src.get("case_name_full")
        or src.get("case_name_short")
        or ""
    ).strip()


def render(src):
    warnings = []

    name = _name(src)
    if not name:
        warnings.append("no case name")

    pd = dates.parse(src.get("date_filed", ""))
    year = dates.year_str(pd)
    if not year:
        warnings.append("no date — year omitted")

    reporter = cit.first_normalized(src.get("citations", []))
    if not reporter:
        warnings.append("no reporter cite")

    url = (src.get("cluster_url") or "").strip()

    parts = []
    if name:
        parts.append(name)
    if year:
        parts.append(f"({year}).")
    elif name:
        # no year: just terminate the name with a period
        parts[-1] = parts[-1] + "."
    if reporter:
        parts.append(f"{reporter}.")
    if url:
        parts.append(url)

    return {
        "claim_id": src.get("claim_id", ""),
        "doc_id":   str(src.get("doc_id", "")),
        "style":    "apa",
        "citation": " ".join(parts).strip(),
        "warnings": warnings,
    }
