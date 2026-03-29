# Extract and normalize metadata from raw CourtListener JSON files.

# TEST
#     python -m src.ingest.collect_metadata
#     python -m src.ingest.collect_metadata --input data/raw/opinions --output data/processed/metadata

# Reads cluster JSON files saved by fetch_courtlistener and produces:
#   - One metadata JSON per document  (metadata/<cluster_id>.json)
#   - A combined catalog file          (metadata/catalog.jsonl)

import argparse
import json
import logging
import re
from pathlib import Path

from src.config import METADATA_DIR, OPINIONS_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# HTML tag stripper (lightweight, no extra dependency)
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _get_nested(mapping: dict, *keys: str, default: str = "") -> str:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    if current is None:
        return default
    return str(current)


def _extract_court_id(cluster: dict, opinions: list[dict]) -> str:
    docket = cluster.get("docket")
    if isinstance(docket, dict):
        value = docket.get("court_id") or docket.get("court")
        if value is not None:
            return str(value)

    for candidate in (
        _get_nested(cluster, "court", "id"),
        _get_nested(cluster, "court_id"),
        _get_nested(cluster, "court"),
    ):
        if candidate:
            return candidate

    for opinion in opinions:
        for candidate in (
            _get_nested(opinion, "court", "id"),
            _get_nested(opinion, "court_id"),
            _get_nested(opinion, "court"),
        ):
            if candidate:
                return candidate

    return ""


def _extract_docket_number(cluster: dict) -> str:
    docket = cluster.get("docket")
    if isinstance(docket, dict):
        value = docket.get("docket_number") or docket.get("docket_number_core")
        if value is not None:
            return str(value)
    return _get_nested(cluster, "docket_number")


# Extraction


def extract_metadata(record: dict) -> dict:
    # Pull structured metadata from a raw cluster+opinions record.

    # Returns a flat dict suitable for indexing / filtering.
    cluster = record.get("cluster", {})
    opinions = record.get("opinions", [])

    # Citations — CourtListener returns these as a list of objects or strings
    raw_citations = _as_list(cluster.get("citations", []))
    citations = []
    for c in raw_citations:
        if isinstance(c, dict):
            citations.append(c.get("cite", str(c)))
        else:
            citations.append(str(c))

    # Judges — may be a string or list
    judges_raw = cluster.get("judges", "") or ""
    if isinstance(judges_raw, list):
        judges = [str(j).strip() for j in judges_raw if str(j).strip()]
    else:
        judges = [j.strip() for j in judges_raw.split(",") if j.strip()]

    # Opinion types present
    opinion_types = []
    for op in opinions:
        op_type = op.get("type", "unknown")
        if op_type not in opinion_types:
            opinion_types.append(op_type)

    # Best available plain text (for word count, preview, etc.)
    full_text = ""
    for op in opinions:
        text = op.get("_text", "") or ""
        if text:
            full_text += strip_html(text) + "\n"

    cluster_id = cluster.get("id")

    return {
        "id": cluster_id,
        "source": "courtlistener",
        "case_name": cluster.get("case_name", ""),
        "case_name_short": cluster.get("case_name_short", ""),
        "case_name_full": cluster.get("case_name_full", ""),
        "date_filed": cluster.get("date_filed", ""),
        "court_id": _extract_court_id(cluster, opinions),
        "docket_number": _extract_docket_number(cluster),
        "citations": citations,
        "judges": judges,
        "opinion_types": opinion_types,
        "num_opinions": len(opinions),
        "word_count": len(full_text.split()),
        "preview": full_text[:500],
        "cluster_url": f"https://www.courtlistener.com/opinion/{cluster_id}/",
    }


#Pipeline 


def process_all(
    input_dir: Path = OPINIONS_RAW_DIR,
    output_dir: Path = METADATA_DIR,
) -> list[dict]:
    # Read every cluster_*.json in *input_dir*, extract metadata, and write individual + catalog files to *output_dir*.
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(input_dir.glob("cluster_*.json"))

    if not raw_files:
        log.warning("No cluster files found in %s", input_dir)
        return []

    catalog: list[dict] = []

    for path in raw_files:
        log.info("Processing %s …", path.name)
        record = json.loads(path.read_text(encoding="utf-8"))
        meta = extract_metadata(record)

        # Save individual metadata file
        dest = output_dir / f"{path.stem}_meta.json"
        dest.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        catalog.append(meta)

    # Write combined catalog (one JSON object per line)
    catalog_path = output_dir / "catalog.jsonl"
    with open(catalog_path, "w", encoding="utf-8") as f:
        for entry in catalog:
            f.write(json.dumps(entry, default=str) + "\n")

    log.info(
        "Done. Processed %d files → %s  (catalog: %s)",
        len(catalog), output_dir, catalog_path,
    )
    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from raw CourtListener JSON files."
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=OPINIONS_RAW_DIR,
        help="Directory with raw cluster JSON files",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=METADATA_DIR,
        help="Directory to write metadata files",
    )
    args = parser.parse_args()
    process_all(input_dir=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
