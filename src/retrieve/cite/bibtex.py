# BibTeX entry for a legal case. Using @misc since there's no @case in
# vanilla BibTeX (biblatex has @jurisdiction but we're not assuming that).

import re
from . import citations as cit
from . import dates


_NON_KEY = re.compile(r"[^A-Za-z0-9]+")


def _bibkey(src):
    doc_id = str(src.get("doc_id", "")).strip()
    if doc_id:
        return "cluster_" + _NON_KEY.sub("", doc_id)

    name = (src.get("case_name_short") or src.get("case_name") or "").strip()
    yr = dates.year_str(dates.parse(src.get("date_filed", "")))
    slug = _NON_KEY.sub("", name) or "case"
    return slug + yr if yr else slug


def _esc(s):
    # bare minimum BibTeX escaping
    return s.replace("\\", "\\\\").replace("{", r"\{").replace("}", r"\}")


def render(src):
    warnings = []

    title = (src.get("case_name")
             or src.get("case_name_full")
             or src.get("case_name_short")
             or "").strip()
    if not title:
        warnings.append("no case name; title left blank")

    pd = dates.parse(src.get("date_filed", ""))
    year = dates.year_str(pd)
    if not year:
        warnings.append("no date — year omitted")

    url = (src.get("cluster_url") or "").strip()
    reporter = cit.first_normalized(src.get("citations", []))

    fields = []
    if title:    fields.append(("title", _esc(title)))
    if year:     fields.append(("year",  year))
    if url:      fields.append(("url",   url))
    if reporter: fields.append(("note",  _esc(reporter)))

    key = _bibkey(src)
    if fields:
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
        entry = f"@misc{{{key},\n{body}\n}}"
    else:
        entry = f"@misc{{{key}}}"

    return {
        "claim_id": src.get("claim_id", ""),
        "doc_id":   str(src.get("doc_id", "")),
        "style":    "bibtex",
        "citation": entry,
        "warnings": warnings,
    }
