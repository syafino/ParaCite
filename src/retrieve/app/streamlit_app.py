"""Streamlit frontend for ParaCite."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieve.app.api import run_citation_pipeline  # noqa: E402


STYLE_OPTIONS = ["bluebook", "apa", "mla", "ieee", "bibtex"]
SAMPLE_TEXT = (
    "Public schools may not segregate students by race. "
    "A suspect must be informed of certain rights before custodial interrogation."
)


def main() -> None:
    st.set_page_config(
        page_title="ParaCite",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ParaCite")
    st.caption("Find candidate legal citations for claims in your writing.")

    with st.sidebar:
        st.header("Citation Settings")
        style = st.selectbox("Citation style", STYLE_OPTIONS, index=0)
        top_k = st.slider("Candidates per claim", min_value=1, max_value=5, value=3)
        st.info(
            "The frontend is wired to a stable adapter. If parser/retriever/"
            "formatter modules are missing, ParaCite shows deterministic demo results."
        )

    text = st.text_area(
        "Paste text to cite",
        height=260,
        placeholder="Paste a paragraph or draft section here...",
        key="input_text",
    )

    left, right = st.columns([1, 4])
    with left:
        run_clicked = st.button("Find citations", type="primary", use_container_width=True)
    with right:
        if st.button("Use sample text"):
            st.session_state["input_text"] = SAMPLE_TEXT
            st.rerun()

    if not run_clicked:
        st.write("Paste text, choose a style, then run ParaCite.")
        return

    result = run_citation_pipeline(text=text, style=style, top_k=top_k)
    _render_status(result)
    _render_results(result)


def _render_status(result: dict) -> None:
    status = result.get("status")
    message = result.get("message", "")
    if status == "ok":
        st.success(message or "Citations generated.")
    elif status == "demo":
        st.warning(message or "Demo citations shown because backend modules are unavailable.")
    else:
        st.error(message or "Unable to generate citations.")


def _render_results(result: dict) -> None:
    claims = result.get("claims") or []
    if not claims:
        st.info("No citation suggestions to display yet.")
        return

    st.subheader("Citation Suggestions")
    for claim in claims:
        with st.expander(f"{claim.get('claim_id', 'claim')}: {claim.get('text', '')[:90]}", expanded=True):
            st.markdown(f"**Claim:** {claim.get('text', '')}")
            st.caption(
                f"Character span: {claim.get('char_start', 0)}-{claim.get('char_end', 0)}"
            )

            citations = claim.get("citations") or []
            if not citations:
                st.info("No citations were returned for this claim.")
                continue

            for rank, citation in enumerate(citations, start=1):
                _render_citation(rank, citation)


def _render_citation(rank: int, citation: dict) -> None:
    st.markdown(f"**Candidate {rank}**")
    language = "bibtex" if citation.get("style") == "bibtex" else "text"
    st.code(citation.get("citation", ""), language=language)

    score = citation.get("score")
    doc_id = citation.get("doc_id") or "unknown"
    details = [f"doc_id: `{doc_id}`", f"style: `{citation.get('style', '')}`"]
    if score is not None:
        details.append(f"score: `{score}`")
    st.markdown(" | ".join(details))

    url = citation.get("cluster_url")
    if url:
        st.markdown(f"[Open source]({url})")

    matched_chunk = citation.get("matched_chunk")
    if matched_chunk:
        with st.expander("Matched snippet"):
            st.write(matched_chunk)

    warnings = citation.get("warnings") or []
    for warning in warnings:
        st.warning(warning)


if __name__ == "__main__":
    main()
