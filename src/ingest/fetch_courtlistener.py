
# Fetch court opinions from the CourtListener REST API (v4).

# TEST
#     python -m src.ingest.fetch_courtlistener --query "Fourth Amendment" --max-pages 3
#     python -m src.ingest.fetch_courtlistener --query "Fourth Amendment" --court scotus --max-pages 2

# Requires COURTLISTENER_TOKEN .env file


import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

from src.config import (
    CL_API_TOKEN,
    CL_BASE_URL,
    CL_DEFAULT_PAGE_SIZE,
    CL_MAX_PAGES,
    CL_MAX_RETRIES,
    CL_REQUEST_TIMEOUT,
    OPINIONS_RAW_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _headers() -> dict:
    if not CL_API_TOKEN:
        log.error(
            "COURTLISTENER_TOKEN not set. "
            "Export it or add it to your environment:\n"
            "  export COURTLISTENER_TOKEN=your_token_here"
        )
        sys.exit(1)
    return {"Authorization": f"Token {CL_API_TOKEN}"}


# GET with retry on timeout/connection errors
def _get_with_retry(url: str, retries: int = CL_MAX_RETRIES, **kwargs) -> requests.Response:
    kwargs.setdefault("headers", _headers())
    kwargs.setdefault("timeout", CL_REQUEST_TIMEOUT)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt < retries:
                wait = attempt * 2
                log.warning("Attempt %d/%d failed (%s). Retrying in %ds …", attempt, retries, exc, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


# Search Endpoint


def search_opinions(
    query: str,
    court: str | None = None,
    date_gte: str | None = None,
    date_lte: str | None = None,
    max_pages: int = CL_MAX_PAGES,
) -> list[dict]:
    # Search CourtListener for opinion clusters matching the query.
    # Returns a flat list of cluster result dicts across up to max_pages.
    params: dict = {"q": query, "type": "o", "page_size": CL_DEFAULT_PAGE_SIZE}
    if court:
        params["court"] = court
    if date_gte:
        params["filed_after"] = date_gte
    if date_lte:
        params["filed_before"] = date_lte

    results: list[dict] = []
    url: str | None = f"{CL_BASE_URL}/search/"
    page = 0

    while url and page < max_pages:
        log.info("Fetching search page %d …", page + 1)
        resp = _get_with_retry(url, params=params if page == 0 else None)
        data = resp.json()
        results.extend(data.get("results", []))
        url = data.get("next")
        page += 1
        time.sleep(0.5)  # be polite

    log.info("Collected %d search results.", len(results))
    return results


# Fetch opinion


def fetch_cluster(cluster_id: int) -> dict:
    # Fetch a single opinion cluster by ID.
    url = f"{CL_BASE_URL}/clusters/{cluster_id}/"
    return _get_with_retry(url).json()


def fetch_opinion(opinion_url: str) -> dict:
    # Fetch a single opinion given its full API URL.
    return _get_with_retry(opinion_url).json()


def get_opinion_text(opinion: dict) -> str:
    # Extract the best available plain-text from an opinion response.

    # CourtListener stores text in several fields; not all are populated.
    # Priority: plain_text > html_with_citations > html > xml_harvard.

    for field in ("plain_text", "html_with_citations", "html", "xml_harvard"):
        text = opinion.get(field)
        if text:
            return text
    return ""


# Fetch & Save


def fetch_and_save(
    query: str,
    court: str | None = None,
    date_gte: str | None = None,
    date_lte: str | None = None,
    max_pages: int = CL_MAX_PAGES,
    out_dir: Path = OPINIONS_RAW_DIR,
) -> list[Path]:
    # Search for opinions, fetch full text for each result, and save raw
    # JSON responses to *out_dir*.

    # Returns list of saved file paths.
    out_dir.mkdir(parents=True, exist_ok=True)

    search_results = search_opinions(
        query, court=court, date_gte=date_gte, date_lte=date_lte,
        max_pages=max_pages,
    )

    saved: list[Path] = []
    for i, result in enumerate(search_results):
        cluster_id = result.get("cluster_id") or result.get("id")
        if not cluster_id:
            log.warning("Skipping result %d – no cluster_id found.", i)
            continue

        log.info(
            "[%d/%d] Fetching cluster %s …",
            i + 1, len(search_results), cluster_id,
        )

        try:
            cluster = fetch_cluster(cluster_id)
        except (requests.HTTPError, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            log.warning("Failed to fetch cluster %s: %s", cluster_id, exc)
            continue

        # Fetch each sub-opinion in the cluster
        opinions = []
        for op_url in cluster.get("sub_opinions", []):
            try:
                opinion = fetch_opinion(op_url)
                opinion["_text"] = get_opinion_text(opinion)
                opinions.append(opinion)
            except (requests.HTTPError, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                log.warning("Failed to fetch opinion %s: %s", op_url, exc)

        # Bundle cluster + opinions together
        record = {
            "cluster": cluster,
            "opinions": opinions,
        }

        dest = out_dir / f"cluster_{cluster_id}.json"
        dest.write_text(json.dumps(record, indent=2, default=str))
        saved.append(dest)
        log.info("  → saved %s", dest.name)

        time.sleep(0.5)

    log.info("Done. Saved %d records to %s", len(saved), out_dir)
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Fetch court opinions from CourtListener."
    )
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--court", "-c", default=None, help="Court ID filter (e.g. scotus)")
    parser.add_argument("--date-gte", default=None, help="Filed on or after (YYYY-MM-DD)")
    parser.add_argument("--date-lte", default=None, help="Filed on or before (YYYY-MM-DD)")
    parser.add_argument(
        "--max-pages", type=int, default=CL_MAX_PAGES,
        help=f"Max search pages to fetch (default {CL_MAX_PAGES})",
    )
    args = parser.parse_args()

    fetch_and_save(
        query=args.query,
        court=args.court,
        date_gte=args.date_gte,
        date_lte=args.date_lte,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
