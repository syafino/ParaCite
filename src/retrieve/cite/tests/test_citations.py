from src.retrieve.cite import citations


def test_clean_string_passes_through():
    assert citations.normalize("347 U.S. 483") == "347 U.S. 483"


def test_dict_with_volume_reporter_page():
    d = {"volume": "347", "reporter": "U.S.", "page": "483"}
    assert citations.normalize(d) == "347 U.S. 483"


def test_dict_with_cite_field_wins():
    d = {"cite": "347 U.S. 483", "volume": "x"}
    assert citations.normalize(d) == "347 U.S. 483"


def test_stringified_dict_from_ingest_bug():
    # this is the actual shape we get back from ingest right now
    raw = "{'volume': '2020', 'reporter': 'Ohio', 'page': '3012', 'type': 8}"
    assert citations.normalize(raw) == "2020 Ohio 3012"


def test_garbage_inputs():
    assert citations.normalize("") is None
    assert citations.normalize(None) is None
    assert citations.normalize("{not a dict") is None
    assert citations.normalize({}) is None


def test_first_normalized_skips_unparseable():
    cs = ["{garbage", "347 U.S. 483", "ignored"]
    assert citations.first_normalized(cs) == "347 U.S. 483"


def test_first_normalized_empty():
    assert citations.first_normalized([]) is None
    assert citations.first_normalized(None) is None
