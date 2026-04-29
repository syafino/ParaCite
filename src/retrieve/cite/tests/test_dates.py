from src.retrieve.cite import dates


def test_parse_iso():
    d = dates.parse("1954-05-17")
    assert (d.year, d.month, d.day) == (1954, 5, 17)
    assert d.has_full


def test_year_only_falls_back():
    d = dates.parse("1954")
    assert d.year == 1954
    assert d.month is None and d.day is None


def test_empty():
    assert dates.parse("").year is None


def test_garbage():
    assert dates.parse("not-a-date").year is None


def test_year_str():
    assert dates.year_str(dates.parse("2020-05-20")) == "2020"
    assert dates.year_str(dates.parse("")) == ""


def test_mla_long():
    assert dates.mla_long(dates.parse("1954-05-17")) == "17 May 1954"
    assert dates.mla_long(dates.parse("1954")) == "1954"


def test_apa_long():
    assert dates.apa_long(dates.parse("1954-05-17")) == "1954, May 17"
    assert dates.apa_long(dates.parse("1954")) == "1954"
