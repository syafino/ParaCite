import pytest
from src.retrieve.cite import AVAILABLE_STYLES, format_citation


# A clean SCOTUS case — everything filled in
BROWN = {
    "claim_id": "c0",
    "doc_id": "12345",
    "source": "courtlistener",
    "case_name": "Brown v. Board of Education",
    "case_name_short": "Brown v. Bd. of Educ.",
    "case_name_full": "Brown v. Board of Education of Topeka",
    "date_filed": "1954-05-17",
    "court_id": "scotus",
    "docket_number": "1",
    "citations": ["347 U.S. 483"],
    "judges": ["Warren"],
    "cluster_url": "https://www.courtlistener.com/opinion/12345/",
}

# Real shape from data/processed/metadata — note the malformed citation
# string and the empty court_id
LEATHERWOOD = {
    "claim_id": "c1",
    "doc_id": "4755117",
    "source": "courtlistener",
    "case_name": "State v. Leatherwood",
    "case_name_short": "Leatherwood",
    "case_name_full": "",
    "date_filed": "2020-05-20",
    "court_id": "",
    "docket_number": "",
    "citations": [
        "{'date_created': '2025-09-25', 'volume': '2020', 'reporter': 'Ohio', 'page': '3012', 'type': 8}"
    ],
    "judges": ["Callahan"],
    "cluster_url": "https://www.courtlistener.com/opinion/4755117/",
}

EMPTY = {"claim_id": "c2", "doc_id": "999"}


def test_styles_listed():
    assert AVAILABLE_STYLES == ["apa", "bibtex", "bluebook", "mla"]


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        format_citation(BROWN, "chicago")


# --- Bluebook ---------------------------------------------------------

def test_bluebook_brown():
    out = format_citation(BROWN, "bluebook")
    assert out["citation"] == "Brown v. Bd. of Educ., 347 U.S. 483 (U.S. 1954)."
    assert out["warnings"] == []
    assert out["style"] == "bluebook"
    assert out["claim_id"] == "c0"


def test_bluebook_recovers_stringified_citation():
    out = format_citation(LEATHERWOOD, "bluebook")
    assert out["citation"] == "Leatherwood, 2020 Ohio 3012 (2020)."
    assert out["warnings"] == []


def test_bluebook_empty_warns_about_everything():
    out = format_citation(EMPTY, "bluebook")
    assert out["citation"] == ""
    # should warn about the three missing pieces
    blob = " | ".join(out["warnings"]).lower()
    assert "case name" in blob
    assert "reporter" in blob
    assert "date" in blob


# --- APA --------------------------------------------------------------

def test_apa_brown():
    out = format_citation(BROWN, "apa")
    assert out["citation"] == (
        "Brown v. Board of Education (1954). 347 U.S. 483. "
        "https://www.courtlistener.com/opinion/12345/"
    )
    assert out["warnings"] == []


# --- MLA --------------------------------------------------------------

def test_mla_brown():
    out = format_citation(BROWN, "mla")
    assert out["citation"] == (
        "Brown v. Board of Education. "
        "Supreme Court of the United States, 17 May 1954, "
        "https://www.courtlistener.com/opinion/12345/."
    )
    assert out["warnings"] == []


# --- BibTeX -----------------------------------------------------------

def test_bibtex_brown():
    out = format_citation(BROWN, "bibtex")
    assert out["style"] == "bibtex"
    s = out["citation"]
    assert s.startswith("@misc{cluster_12345,")
    assert "title = {Brown v. Board of Education}" in s
    assert "year = {1954}" in s
    assert "url = {https://www.courtlistener.com/opinion/12345/}" in s
    assert "note = {347 U.S. 483}" in s
    assert s.endswith("}")


def test_bibtex_minimal_when_empty():
    out = format_citation(EMPTY, "bibtex")
    assert out["citation"] == "@misc{cluster_999}"
