"""Chunk the catalog produced by collect_metadata.py.

Reads data/processed/metadata/catalog.jsonl and produces data/processed/text/chunks.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import METADATA_DIR, TEXT_DIR
from src.ingest.chunk_text import chunk_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    catalog_path = METADATA_DIR / "catalog.jsonl"
    if not catalog_path.exists():
        log.error("Catalog not found at %s. Run collect_metadata first.", catalog_path)
        return

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TEXT_DIR / "chunks.jsonl"

    log.info("Reading catalog from %s", catalog_path)
    all_chunks = []
    
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            doc_id = record.get("id")
            text = record.get("text", "")
            
            if not doc_id or not text:
                continue
                
            meta = {
                "case_name": record.get("case_name"),
                "court_id": record.get("court_id"),
                "date_filed": record.get("date_filed"),
                "cluster_url": record.get("cluster_url"),
            }
            
            chunks = chunk_text(str(doc_id), text, extra_metadata=meta)
            all_chunks.extend(chunks)

    log.info("Chunked %d documents into %d total chunks", len(all_chunks), len(all_chunks))
    
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
            
    log.info("Wrote chunks to %s", output_path)


if __name__ == "__main__":
    main()
