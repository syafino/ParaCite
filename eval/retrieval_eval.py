from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_QUERIES_PATH = Path(__file__).resolve().parent / "eval_queries.jsonl"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "metrics.json"
DEFAULT_DETAILS_PATH = Path(__file__).resolve().parent / "results" / "retrieval_details.jsonl"


def load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"eval queries not found: {path}")

    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            row = json.loads(line)
            query = str(row.get("query") or "").strip()
            relevant_doc_ids = [str(doc_id) for doc_id in row.get("relevant_doc_ids") or []]
            if not query or not relevant_doc_ids:
                raise ValueError(
                    f"{path}:{line_number} must include query and relevant_doc_ids"
                )

            queries.append(
                {
                    "query_id": str(row.get("query_id") or f"q{line_number}"),
                    "query": query,
                    "relevant_doc_ids": relevant_doc_ids,
                }
            )

    if not queries:
        raise ValueError(f"no eval queries found in {path}")
    return queries


def unique_doc_ids(results: list[dict[str, Any]]) -> list[str]:
    # Retrieval returns chunks. For this eval we grade at the paper/doc level.
    seen: set[str] = set()
    ranked: list[str] = []

    for result in results:
        doc_id = str(result.get("doc_id") or "")
        if not doc_id or doc_id in seen:
            continue

        seen.add(doc_id)
        ranked.append(doc_id)

    return ranked


def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0

    retrieved = set(ranked_doc_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(ranked_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    relevant = set(relevant_doc_ids)
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    queries: list[dict[str, Any]],
    top_k: int,
    recall_ks: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.app import get_hybrid_search

    search = get_hybrid_search()
    details: list[dict[str, Any]] = []
    recalls: dict[int, list[float]] = {k: [] for k in recall_ks}
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for query in queries:
        # Keep timing around just the search call. Model/index load happens before this.
        started = perf_counter()
        results = search.query(query["query"], top_k=top_k)
        elapsed_ms = (perf_counter() - started) * 1000.0

        ranked_doc_ids = unique_doc_ids(results)
        rr = reciprocal_rank(ranked_doc_ids, query["relevant_doc_ids"])
        reciprocal_ranks.append(rr)
        latencies_ms.append(elapsed_ms)

        per_query_recalls = {}

        for k in recall_ks:
            value = recall_at_k(ranked_doc_ids, query["relevant_doc_ids"], k)
            recalls[k].append(value)
            per_query_recalls[f"recall@{k}"] = value

        details.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "relevant_doc_ids": query["relevant_doc_ids"],
                "ranked_doc_ids": ranked_doc_ids[:top_k],
                "reciprocal_rank": rr,
                "latency_ms": elapsed_ms,
                **per_query_recalls,
            }
        )

    metrics: dict[str, Any] = {
        "num_queries": len(queries),
        "top_k": top_k,
        "mrr": mean(reciprocal_ranks),
        "mean_latency_ms": mean(latencies_ms),
    }

    for k in recall_ks:
        metrics[f"recall@{k}"] = mean(recalls[k])

    return metrics, details


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_ks(raw: str) -> list[int]:
    # accepts --ks 1,3,5,10
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("ks must be comma-separated positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ParaCite retrieval with Recall@K and MRR")

    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", type=parse_ks, default=parse_ks("1,3,5,10"))

    args = parser.parse_args()

    max_k = max(args.ks)
    top_k = max(args.top_k, max_k)

    queries = load_queries(args.queries)
    metrics, details = evaluate(queries, top_k=top_k, recall_ks=args.ks)

    write_json(args.output, metrics)
    write_jsonl(args.details, details)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
