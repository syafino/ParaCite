# Bluebook (Bluepages B10) — the standard for legal cases.
# Pattern:  <Case Name>, <reporter cite> (<court abbrev> <year>).
# Example:  Brown v. Bd. of Educ., 347 U.S. 483 (U.S. 1954).

from . import citations as cit
from . import courts
from . import dates


def _case_name(src):
    # short form is preferred for Bluebook, fall back if missing
    return (
        src.get("case_name_short")
        or src.get("case_name")
        or src.get("case_name_full")
        or ""
    ).strip()


def render(src):
    warns = []

    name = _case_name(src)
    if not name:
        warns.append("no case name")

    reporter = cit.first_normalized(src.get("citations", []))
    if not reporter:
        warns.append("no reporter cite")

    pd = dates.parse(src.get("date_filed", ""))
    year = dates.year_str(pd)
    if not year:
        warns.append("no date — year omitted")

    court = courts.bluebook_abbrev(src.get("court_id", ""))
    paren = " ".join(p for p in (court, year) if p)

    out = []
    if name:
        out.append(f"{name},")
    if reporter:
        out.append(reporter)
    if paren:
        out.append(f"({paren})")

    citation = " ".join(out).strip()
    if citation:
        citation = citation.rstrip(".") + "."

    return {
        "claim_id": src.get("claim_id", ""),
        "doc_id":   str(src.get("doc_id", "")),
        "style":    "bluebook",
        "citation": citation,
        "warnings": warns,
    }
