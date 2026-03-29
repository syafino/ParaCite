import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import INDEX_DIR, TEXT_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

VECTORS_DIR = INDEX_DIR / "vectors"
DEFAULT_CHUNKS_PATH = TEXT_DIR / "chunks.jsonl"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunks_path: Path) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    records: list[dict] = []
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
            records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": metadata,
                }
            )

    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_vectors(
    chunks_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> None:
    records = load_chunks(chunks_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing dependency `numpy`. Install it before running this script.") from exc

    if not records:
        empty_matrix = np.empty((0, 0), dtype=np.float32)
        np.save(output_dir / "embeddings.npy", empty_matrix)
        write_jsonl(output_dir / "records.jsonl", [])
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(chunks_path),
            "model_name": model_name,
            "normalize_embeddings": normalize_embeddings,
            "batch_size": batch_size,
            "num_chunks": 0,
            "embedding_dim": 0,
            "artifacts": {
                "embeddings": "embeddings.npy",
                "records": "records.jsonl",
            },
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("No chunks found in %s. Wrote empty vector index to %s", chunks_path, output_dir)
        return

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `sentence-transformers`. Install it before running this script."
        ) from exc

    model = SentenceTransformer(model_name)
    texts = [record["text"] for record in records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = embeddings.astype(np.float32, copy=False)

    np.save(output_dir / "embeddings.npy", embeddings)
    write_jsonl(output_dir / "records.jsonl", records)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(chunks_path),
        "model_name": model_name,
        "normalize_embeddings": normalize_embeddings,
        "batch_size": batch_size,
        "num_chunks": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "artifacts": {
            "embeddings": "embeddings.npy",
            "records": "records.jsonl",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote vector index for %d chunks to %s", len(records), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sentence embeddings for chunked text.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the JSONL chunk corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VECTORS_DIR,
        help="Directory where vector artifacts will be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name to use for encoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used while encoding text chunks.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization on the generated embeddings.",
    )
    args = parser.parse_args()

    build_vectors(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
