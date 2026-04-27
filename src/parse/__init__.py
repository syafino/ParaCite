"""Sentence parsing: turn raw user text into Claim dicts (§3.1).

Public API::

    from src.parse import extract_claims
    claims = extract_claims("Some text...")
"""

from src.parse.claims import extract_claims

__all__ = ["extract_claims"]
