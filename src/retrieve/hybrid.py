"""Hybrid retrieval for ParaCite.
BM25 keyword matching over chunk text
sentence-transformer semantic retrieval via the existing FAISS store
optional word2vec semantic retrieval when a word2vec FAISS index exists

Scores from each available retriever are normalized per query and combined
with configurable weights. Missing optional indexes are skipped, so the app can
still run with only the sentence-transformer index.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embeddings.base import Embedder
from src.index.vector_store import VECTORS_DIR, VectorStore
from src.index.build_bm25 import BM25_DIR, build_bm25_payload, tokenize
from src.retrieve.search import SemanticSearch

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "bm25": 0.40,
    "sentence_transformer": 0.45,
    "word2vec": 0.15,
}


@dataclass(frozen=True)
class HybridWeights:
    bm25: float = DEFAULT_WEIGHTS["bm25"]
    sentence_transformer: float = DEFAULT_WEIGHTS["sentence_transformer"]
    word2vec: float = DEFAULT_WEIGHTS["word2vec"]


class BM25Index:
    """Small in-process BM25 scorer loaded from disk or built from records."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.params = payload.get("params") or {}
        self.k1 = float(self.params.get("k1", 1.5))
        self.b = float(self.params.get("b", 0.75))
        self.idf = {str(k): float(v) for k, v in (payload.get("idf") or {}).items()}
        self.documents = list(payload.get("documents") or [])
        stats = payload.get("stats") or {}
        self.avg_doc_len = float(stats.get("avg_doc_len") or 0.0)
        self.num_documents = len(self.documents)

    @classmethod
    def load(cls, index_dir: Path = BM25_DIR) -> "BM25Index":
        payload_path = index_dir / "index.json"
        with payload_path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "BM25Index":
        chunk_records = []
        for record in records:
            chunk_records.append(
                _ChunkLike(
                    chunk_id=str(record.get("chunk_id", "")),
                    doc_id=str(record.get("doc_id", "")),
                    text=str(record.get("text", "")),
                    metadata=dict(record.get("metadata") or {}),
                )
            )
        return cls(build_bm25_payload(chunk_records, k1=1.5, b=0.75))

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_terms = Counter(tokenize(query))
        if not query_terms or not self.documents:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for document in self.documents:
            score = self._score_document(query_terms, document)
            if score <= 0:
                continue
            scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, document in scored[:top_k]:
            results.append(
                {
                    "score": float(score),
                    "chunk_id": document.get("chunk_id", ""),
                    "doc_id": document.get("doc_id", ""),
                    "text": document.get("text", ""),
                    "metadata": document.get("metadata") or {},
                }
            )
        return results

    def _score_document(self, query_terms: Counter[str], document: dict[str, Any]) -> float:
        term_freqs = document.get("term_freqs") or {}
        doc_len = float(document.get("length") or 0.0)
        if doc_len <= 0 or self.avg_doc_len <= 0:
            return 0.0

        score = 0.0
        length_norm = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
        for term, query_count in query_terms.items():
            tf = float(term_freqs.get(term) or 0.0)
            if tf <= 0:
                continue
            numerator = tf * (self.k1 + 1.0)
            score += query_count * self.idf.get(term, 0.0) * (numerator / (tf + length_norm))
        return score


@dataclass
class _ChunkLike:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


class HybridSearch:
    """Combine keyword, sentence-transformer, and optional word2vec results."""

    def __init__(
        self,
        sentence_search: SemanticSearch,
        bm25_index: BM25Index | None = None,
        word2vec_search: SemanticSearch | None = None,
        weights: HybridWeights = HybridWeights(),
    ) -> None:
        self.sentence_search = sentence_search
        self.bm25_index = bm25_index
        self.word2vec_search = word2vec_search
        self.weights = weights

    @classmethod
    def from_sentence_store(
        cls,
        sentence_embedder: Embedder,
        sentence_store: VectorStore,
        weights: HybridWeights = HybridWeights(),
    ) -> "HybridSearch":
        sentence_search = SemanticSearch(sentence_embedder, sentence_store)
        bm25_index = _load_or_build_bm25(sentence_store)
        word2vec_search = _load_word2vec_search()
        return cls(
            sentence_search=sentence_search,
            bm25_index=bm25_index,
            word2vec_search=word2vec_search,
            weights=weights,
        )

    def query(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.search(text, top_k=top_k)

    def search(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._refresh_bm25_if_store_changed()
        safe_top_k = max(1, int(top_k or 1))
        pool_size = max(safe_top_k * 4, 20)
        channels: list[tuple[str, float, list[dict[str, Any]]]] = []

        if self.bm25_index is not None:
            channels.append(("bm25", self.weights.bm25, self.bm25_index.search(text, top_k=pool_size)))

        sentence_hits = self.sentence_search.query(text, top_k=pool_size)
        channels.append(("sentence_transformer", self.weights.sentence_transformer, sentence_hits))

        if self.word2vec_search is not None:
            channels.append(("word2vec", self.weights.word2vec, self.word2vec_search.query(text, top_k=pool_size)))

        available_weight = sum(weight for _, weight, hits in channels if hits)
        if available_weight <= 0:
            return []

        merged: dict[str, dict[str, Any]] = {}
        for channel_name, channel_weight, hits in channels:
            normalized_weight = channel_weight / available_weight
            for hit, normalized_score in zip(hits, _normalize_scores([h.get("score", 0.0) for h in hits])):
                key = _hit_key(hit)
                entry = merged.setdefault(
                    key,
                    {
                        **hit,
                        "score": 0.0,
                        "scores": {},
                        "retrieval_modes": [],
                    },
                )
                entry["score"] += normalized_weight * normalized_score
                entry["scores"][channel_name] = {
                    "raw": float(hit.get("score") or 0.0),
                    "normalized": normalized_score,
                }
                entry["retrieval_modes"].append(channel_name)

        results = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        for result in results:
            result["score"] = float(result["score"])
            result["retrieval_modes"] = sorted(set(result["retrieval_modes"]))
        return results[:safe_top_k]

    def _refresh_bm25_if_store_changed(self) -> None:
        if self.bm25_index is None:
            return
        store_records = self.sentence_search.store.records
        if len(store_records) != self.bm25_index.num_documents:
            log.info("Refreshing transient BM25 index after ingest changed the vector store")
            self.bm25_index = BM25Index.from_records(store_records)


def retrieve_for_claims(claims: list[dict[str, Any]], top_k: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Frontend adapter hook used by ``src.retrieve.app.api``."""
    from src.app import get_hybrid_search

    search = get_hybrid_search()
    return {
        str(claim.get("claim_id") or idx): search.query(str(claim.get("text", "")), top_k=top_k)
        for idx, claim in enumerate(claims)
    }


def _load_or_build_bm25(sentence_store: VectorStore) -> BM25Index | None:
    try:
        return BM25Index.load(BM25_DIR)
    except FileNotFoundError:
        if sentence_store.records:
            log.info("No BM25 index found at %s; building transient BM25 from vector records", BM25_DIR)
            return BM25Index.from_records(sentence_store.records)
        return None
    except Exception as exc:  # noqa: BLE001 - optional retrieval channel
        log.warning("BM25 index could not be loaded: %s", exc)
        return None


def _load_word2vec_search() -> SemanticSearch | None:
    word2vec_dir = VECTORS_DIR / "word2vec"
    if not (word2vec_dir / "records.jsonl").exists():
        return None

    try:
        from src.embeddings.word2vec import Word2VecEmbedder

        store = VectorStore.load(word2vec_dir)
        embedder = Word2VecEmbedder()
        return SemanticSearch(embedder, store)
    except Exception as exc:  # noqa: BLE001 - optional retrieval channel
        log.warning("Word2Vec retrieval disabled: %s", exc)
        return None


def _normalize_scores(scores: list[Any]) -> list[float]:
    numeric = [float(score or 0.0) for score in scores]
    if not numeric:
        return []
    lo = min(numeric)
    hi = max(numeric)
    if math.isclose(hi, lo):
        return [1.0 if score > 0 else 0.0 for score in numeric]
    return [(score - lo) / (hi - lo) for score in numeric]


def _hit_key(hit: dict[str, Any]) -> str:
    chunk_id = str(hit.get("chunk_id") or "")
    if chunk_id:
        return f"chunk:{chunk_id}"
    doc_id = str(hit.get("doc_id") or "")
    text = str(hit.get("text") or "")
    return f"fallback:{doc_id}:{hash(text)}"
