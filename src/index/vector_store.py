# Build and query a FAISS-backed vector index from chunked text.
#
# BUILD
#     python -m src.index.vector_store --embedder word2vec
#     python -m src.index.vector_store --embedder sentence-transformer
#     python -m src.index.vector_store --chunks path/to/chunks.jsonl --embedder word2vec
#
# Prerequisite: chunks.jsonl must exist.  If it does not, you will be
# prompted to run the chunking step first.

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from src.config import INDEX_DIR, TEXT_DIR
from src.embeddings.base import Embedder

log = logging.getLogger(__name__)

VECTORS_DIR = INDEX_DIR / "vectors"
DEFAULT_CHUNKS_PATH = TEXT_DIR / "chunks.jsonl"


def _load_chunks(chunks_path: Path) -> list[dict]:
    """Read chunks.jsonl and return a list of record dicts."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    records: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
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
                or f"chunk-{lineno}"
            )
            doc_id = str(
                payload.get("doc_id")
                or payload.get("paper_id")
                or payload.get("document_id")
                or payload.get("source_id")
                or chunk_id
            )
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in {
                    "chunk_id", "id", "chunk",
                    "doc_id", "paper_id", "document_id", "source_id",
                    "text",
                }
            }
            records.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata,
            })
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class VectorStore:
    """FAISS-backed vector index for cosine similarity search.

    Use :meth:`build` to create a new index from chunks + an embedder, or
    :meth:`load` to restore one from disk.
    """

    def __init__(self, index: faiss.Index, records: list[dict]) -> None:
        self.index = index
        self.records = records

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        embedder: Embedder,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        output_dir: Path = VECTORS_DIR,
    ) -> VectorStore:
        """Embed all chunks and persist a FAISS index + metadata to disk."""
        records = _load_chunks(chunks_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not records:
            empty = np.empty((0, 0), dtype=np.float32)
            np.save(output_dir / "embeddings.npy", empty)
            _write_jsonl(output_dir / "records.jsonl", [])
            idx = faiss.IndexFlatIP(0)
            faiss.write_index(idx, str(output_dir / "index.faiss"))
            cls._write_manifest(output_dir, chunks_path, embedder, 0, 0)
            log.info("No chunks found — wrote empty index to %s", output_dir)
            return cls(idx, [])

        texts = [r["text"] for r in records]
        embeddings = _l2_normalize(embedder.embed(texts))

        idx = faiss.IndexFlatIP(embeddings.shape[1])
        idx.add(embeddings)

        faiss.write_index(idx, str(output_dir / "index.faiss"))
        np.save(output_dir / "embeddings.npy", embeddings)
        _write_jsonl(output_dir / "records.jsonl", records)
        cls._write_manifest(
            output_dir, chunks_path, embedder,
            len(records), int(embeddings.shape[1]),
        )
        log.info(
            "Built FAISS index for %d chunks (%d-d) → %s",
            len(records), embeddings.shape[1], output_dir,
        )
        return cls(idx, records)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_manifest(
        output_dir: Path,
        chunks_path: Path,
        embedder: Embedder,
        num_chunks: int,
        embedding_dim: int,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(chunks_path),
            "embedder": type(embedder).__name__,
            "embedding_dim": embedding_dim,
            "num_chunks": num_chunks,
            "artifacts": {
                "faiss_index": "index.faiss",
                "embeddings": "embeddings.npy",
                "records": "records.jsonl",
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path = VECTORS_DIR) -> VectorStore:
        """Load a previously-built vector store from *index_dir*.

        If ``index.faiss`` is missing but ``embeddings.npy`` exists (e.g. an
        index produced by the older ``build_vectors.py``), the FAISS index is
        rebuilt on the fly.
        """
        records = _read_jsonl(index_dir / "records.jsonl")

        faiss_path = index_dir / "index.faiss"
        npy_path = index_dir / "embeddings.npy"

        if faiss_path.exists():
            idx = faiss.read_index(str(faiss_path))
        elif npy_path.exists():
            embeddings = np.load(npy_path).astype(np.float32)
            if embeddings.size == 0:
                idx = faiss.IndexFlatIP(0)
            else:
                embeddings = _l2_normalize(embeddings)
                idx = faiss.IndexFlatIP(embeddings.shape[1])
                idx.add(embeddings)
            log.info("Rebuilt FAISS index from embeddings.npy (%d vectors)", idx.ntotal)
        else:
            raise FileNotFoundError(
                f"Neither index.faiss nor embeddings.npy found in {index_dir}"
            )

        return cls(idx, records)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[dict]:
        """Return the *top_k* nearest records by cosine similarity.

        *query_vec* should be a 1-D float32 array (already L2-normalised).
        """
        query = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, len(self.records)))
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), **self.records[idx]})
        return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _resolve_embedder(name: str, pretrained: str | None, model: str | None) -> Embedder:
    if name == "word2vec":
        from src.embeddings.word2vec import Word2VecEmbedder
        return Word2VecEmbedder(pretrained=pretrained or "glove-wiki-gigaword-100")
    if name == "sentence-transformer":
        from src.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedder: {name!r}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Build a FAISS vector index from chunked text.",
    )
    parser.add_argument(
        "--chunks", type=Path, default=DEFAULT_CHUNKS_PATH,
        help="Path to the JSONL chunk corpus (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=VECTORS_DIR,
        help="Directory for vector index artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--embedder", choices=["word2vec", "sentence-transformer"],
        default="word2vec",
        help="Which embedding backend to use (default: %(default)s).",
    )
    parser.add_argument(
        "--pretrained", default=None,
        help="Gensim model name for word2vec (default: glove-wiki-gigaword-100).",
    )
    parser.add_argument(
        "--model", default=None,
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2).",
    )
    args = parser.parse_args()

    if not args.chunks.exists():
        print(
            f"\nError: Chunk file not found at {args.chunks}\n"
            "\nThe vector index requires a chunked corpus.  Make sure you have\n"
            "run the ingestion + chunking pipeline first.  Typical steps:\n"
            "\n"
            "  1. python -m src.ingest.fetch_courtlistener --query \"...\" --max-pages 3\n"
            "  2. python -m src.ingest.collect_metadata\n"
            "  3. (chunking step that produces chunks.jsonl)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    embedder = _resolve_embedder(args.embedder, args.pretrained, args.model)
    VectorStore.build(embedder, chunks_path=args.chunks, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
