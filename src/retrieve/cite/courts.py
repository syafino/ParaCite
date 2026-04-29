# court_id -> (long name, Bluebook abbreviation)
# Mostly federal courts for now. TODO add the big state ones (CA, NY, TX, OH...)

_COURT_NAMES = {
    "scotus": ("Supreme Court of the United States", "U.S."),
    "ca1":  ("U.S. Court of Appeals for the First Circuit",   "1st Cir."),
    "ca2":  ("U.S. Court of Appeals for the Second Circuit",  "2d Cir."),
    "ca3":  ("U.S. Court of Appeals for the Third Circuit",   "3d Cir."),
    "ca4":  ("U.S. Court of Appeals for the Fourth Circuit",  "4th Cir."),
    "ca5":  ("U.S. Court of Appeals for the Fifth Circuit",   "5th Cir."),
    "ca6":  ("U.S. Court of Appeals for the Sixth Circuit",   "6th Cir."),
    "ca7":  ("U.S. Court of Appeals for the Seventh Circuit", "7th Cir."),
    "ca8":  ("U.S. Court of Appeals for the Eighth Circuit",  "8th Cir."),
    "ca9":  ("U.S. Court of Appeals for the Ninth Circuit",   "9th Cir."),
    "ca10": ("U.S. Court of Appeals for the Tenth Circuit",   "10th Cir."),
    "ca11": ("U.S. Court of Appeals for the Eleventh Circuit", "11th Cir."),
    "cadc": ("U.S. Court of Appeals for the D.C. Circuit",    "D.C. Cir."),
    "cafc": ("U.S. Court of Appeals for the Federal Circuit", "Fed. Cir."),
}


def long_name(court_id):
    if not court_id:
        return ""
    return _COURT_NAMES.get(court_id, (court_id, court_id))[0]


def bluebook_abbrev(court_id):
    if not court_id:
        return ""
    return _COURT_NAMES.get(court_id, (court_id, court_id))[1]
