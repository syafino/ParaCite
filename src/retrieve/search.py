# Semantic search over an existing FAISS vector index.
#
# USAGE
#     python -m src.retrieve.search --query "freedom of speech"
#     python -m src.retrieve.search --embedder sentence-transformer --top-k 3
#     python -m src.retrieve.search          # ← interactive REPL
#
# Prerequisite: a vector index must already exist.  Build one first with
#     python -m src.index.vector_store --embedder word2vec

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.embeddings.base import Embedder
from src.index.vector_store import VectorStore, VECTORS_DIR

log = logging.getLogger(__name__)


class SemanticSearch:
    """High-level semantic search: embed a query, then retrieve from the store."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def query(self, text: str, top_k: int = 10) -> list[dict]:
        """Return the *top_k* chunks most similar to *text*."""
        vec = self.embedder.embed([text])[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return self.store.search(vec, top_k=top_k)


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def _resolve_embedder(name: str, pretrained: str | None, model: str | None) -> Embedder:
    if name == "word2vec":
        from src.embeddings.word2vec import Word2VecEmbedder
        return Word2VecEmbedder(pretrained=pretrained or "glove-wiki-gigaword-100")
    if name == "sentence-transformer":
        from src.embeddings.sentence_transformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedder: {name!r}")


def _print_results(results: list[dict]) -> None:
    if not results:
        print("  (no results)")
        return
    for rank, hit in enumerate(results, start=1):
        score = hit["score"]
        chunk_id = hit.get("chunk_id", "?")
        doc_id = hit.get("doc_id", "?")
        text_preview = hit.get("text", "")[:120].replace("\n", " ")
        print(f"  [{rank}] score={score:.4f}  chunk={chunk_id}  doc={doc_id}")
        print(f"      {text_preview}...")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Search the vector index for chunks similar to a query.",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=VECTORS_DIR,
        help="Directory containing the vector index (default: %(default)s).",
    )
    parser.add_argument(
        "--embedder", choices=["word2vec", "sentence-transformer"],
        default="word2vec",
        help="Embedding backend (must match what was used to build the index).",
    )
    parser.add_argument(
        "--pretrained", default=None,
        help="Gensim model name for word2vec (default: glove-wiki-gigaword-100).",
    )
    parser.add_argument(
        "--model", default=None,
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: %(default)s).",
    )
    parser.add_argument(
        "--query", "-q", default=None,
        help="Query text.  If omitted, starts an interactive REPL.",
    )
    args = parser.parse_args()

    # --- Validate index exists -------------------------------------------
    index_dir = args.index_dir
    if index_dir == VECTORS_DIR and (index_dir / args.embedder).exists():
        index_dir = index_dir / args.embedder
    args.index_dir = index_dir

    records_path = args.index_dir / "records.jsonl"
    faiss_path = args.index_dir / "index.faiss"
    npy_path = args.index_dir / "embeddings.npy"

    if not records_path.exists() and not faiss_path.exists() and not npy_path.exists():
        print(
            f"\nError: No vector index found in {args.index_dir}\n"
            "\nYou need to build the index first.  Run:\n"
            "\n"
            f"  python -m src.index.vector_store --embedder {args.embedder}\n"
            "\n"
            "If you haven't ingested data yet, start with:\n"
            "\n"
            "  1. python -m src.ingest.fetch_courtlistener --query \"...\" --max-pages 3\n"
            "  2. python -m src.ingest.collect_metadata\n"
            "  3. (chunking step that produces chunks.jsonl)\n"
            "  4. python -m src.index.vector_store --embedder word2vec\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load embedder + store -------------------------------------------
    embedder = _resolve_embedder(args.embedder, args.pretrained, args.model)
    store = VectorStore.load(args.index_dir)
    search = SemanticSearch(embedder, store)

    log.info("Loaded index with %d vectors from %s", store.index.ntotal, args.index_dir)

    # --- Single query or interactive REPL --------------------------------
    if args.query:
        results = search.query(args.query, top_k=args.top_k)
        print(f"\nResults for: {args.query!r}\n")
        _print_results(results)
    else:
        print(
            f"\nInteractive search  (index: {args.index_dir}, "
            f"embedder: {args.embedder}, top_k: {args.top_k})\n"
            "Type a query and press Enter.  Ctrl-C or empty line to quit.\n"
        )
        try:
            while True:
                try:
                    text = input("query> ").strip()
                except EOFError:
                    break
                if not text:
                    break
                results = search.query(text, top_k=args.top_k)
                print()
                _print_results(results)
                print()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
