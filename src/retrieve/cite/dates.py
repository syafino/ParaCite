# date helpers used by every formatter
# CourtListener gives us "YYYY-MM-DD" but a lot of records are missing
# the day or even the month, so we have to be defensive.

from dataclasses import dataclass
from datetime import date
from typing import Optional


MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass(frozen=True)
class ParsedDate:
    year: Optional[int]
    month: Optional[int]
    day: Optional[int]
    raw: str

    @property
    def has_year(self):
        return self.year is not None

    @property
    def has_full(self):
        return self.year is not None and self.month is not None and self.day is not None


def parse(s: str) -> ParsedDate:
    s = (s or "").strip()
    if not s:
        return ParsedDate(None, None, None, "")

    try:
        d = date.fromisoformat(s[:10])
        return ParsedDate(d.year, d.month, d.day, s)
    except ValueError:
        pass

    # last-ditch: pull out a year if there's one at the start
    if len(s) >= 4 and s[:4].isdigit():
        return ParsedDate(int(s[:4]), None, None, s)
    return ParsedDate(None, None, None, s)


def year_str(d: ParsedDate) -> str:
    return str(d.year) if d.year is not None else ""


def mla_long(d):
    """e.g. '17 May 1954'. Drops back to month+year, then year, if parts are missing."""
    if d.has_full:
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if d.year and d.month:
        return f"{MONTHS[d.month - 1]} {d.year}"
    return year_str(d)


def apa_long(d):
    # APA wants year first: "1954, May 17"
    if d.has_full:
        return f"{d.year}, {MONTHS[d.month - 1]} {d.day}"
    if d.year and d.month:
        return f"{d.year}, {MONTHS[d.month - 1]}"
    return year_str(d)
