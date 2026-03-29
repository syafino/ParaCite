import argparse
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import INDEX_DIR, TEXT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
BM25_DIR = INDEX_DIR / "bm25"
DEFAULT_CHUNKS_PATH = TEXT_DIR / "chunks.jsonl"


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_chunks(chunks_path: Path) -> list[ChunkRecord]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    records: list[ChunkRecord] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            text = str(payload.get("text", "")).strip()
            if not text:
                continue

            chunk_id = str(
                payload.get("chunk_id")
                or payload.get("id")
                or payload.get("chunk")
                or f"chunk-{line_number}"
            )
            doc_id = str(
                payload.get("doc_id")
                or payload.get("paper_id")
                or payload.get("document_id")
                or payload.get("source_id")
                or chunk_id
            )

            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"chunk_id", "id", "chunk", "doc_id", "paper_id", "document_id", "source_id", "text"}
            }
            records.append(ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, text=text, metadata=metadata))

    return records


def build_bm25_payload(records: list[ChunkRecord], k1: float, b: float) -> dict:
    doc_freqs: Counter[str] = Counter()
    doc_payloads: list[dict] = []
    total_terms = 0

    for record in records:
        tokens = tokenize(record.text)
        total_terms += len(tokens)
        term_freqs = Counter(tokens)
        doc_freqs.update(term_freqs.keys())
        doc_payloads.append(
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "text": record.text,
                "metadata": record.metadata,
                "length": len(tokens),
                "term_freqs": dict(term_freqs),
            }
        )

    num_docs = len(doc_payloads)
    avg_doc_len = (total_terms / num_docs) if num_docs else 0.0
    idf = {
        term: math.log(1.0 + ((num_docs - freq + 0.5) / (freq + 0.5)))
        for term, freq in doc_freqs.items()
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(DEFAULT_CHUNKS_PATH),
        "params": {"k1": k1, "b": b},
        "stats": {
            "num_chunks": num_docs,
            "avg_doc_len": avg_doc_len,
            "total_terms": total_terms,
            "vocab_size": len(idf),
        },
        "idf": idf,
        "documents": doc_payloads,
    }


def write_bm25_index(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "index.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Wrote BM25 index to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BM25 keyword index from chunked text.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the JSONL chunk corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BM25_DIR,
        help="Directory where the BM25 index will be written.",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    args = parser.parse_args()

    records = load_chunks(args.chunks)
    log.info("Loaded %d chunks from %s", len(records), args.chunks)
    payload = build_bm25_payload(records, k1=args.k1, b=args.b)
    payload["source_path"] = str(args.chunks)
    write_bm25_index(payload, args.output_dir)


if __name__ == "__main__":
    main()
