"""Chunk-level retrieval evaluation runner.

For every query in the retrieval set it queries ``/v1/search``, maps the hits to
``(source, chunk_index)`` keys, and scores the ranking with MRR@k, hit-rate@k and
nDCG@k. These are pure retrieval metrics - no LLM or generator is involved, so the
retrieval stage can be tuned and compared in isolation.

Usage:
    python -m rag_platform.evaluation.run_retrieval_eval \\
        --dataset data/retrieval_set.jsonl --api-url http://127.0.0.1:8000 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from rag_platform.config.settings import get_settings
from rag_platform.evaluation.retrieval_metrics import (
    ChunkKey,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
)
from rag_platform.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _search(
    api_url: str,
    query: str,
    top_k: int,
    hybrid: bool,
    rerank: bool,
    timeout: float = 120.0,
) -> list[ChunkKey]:
    """Query /v1/search and return the ranked chunk keys ``(source, chunk_index)``."""
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if hybrid:
        payload["hybrid"] = True
    if rerank:
        payload["rerank"] = True
    response = httpx.post(f"{api_url}/v1/search", json=payload, timeout=timeout)
    response.raise_for_status()
    return [
        (str(hit["metadata"]["source"]), int(hit["metadata"]["chunk_index"]))
        for hit in response.json()["hits"]
    ]


def score_query(
    relevant: list[list[Any]],
    ranked: list[ChunkKey],
    top_k: int,
) -> dict[str, float]:
    """Score one query: MRR@k, hit-rate@k and nDCG@k against the ranked chunks."""
    relevant_keys = {(str(source), int(index)) for source, index in relevant}
    return {
        "mrr_at_k": mrr_at_k(relevant_keys, ranked, top_k),
        "hit_rate_at_k": hit_rate_at_k(relevant_keys, ranked, top_k),
        "ndcg_at_k": ndcg_at_k(relevant_keys, ranked, top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chunk-level retrieval metrics against the live search API."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL retrieval dataset")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="RAG API base URL")
    parser.add_argument("--top-k", default=5, type=int, help="Ranked chunks considered")
    parser.add_argument(
        "--hybrid", action="store_true", help="Use hybrid dense+sparse retrieval"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Rerank candidates with the cross-encoder"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="Log experiment parameters and metrics to MLflow"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    per_query: list[dict[str, Any]] = []
    for row in rows:
        ranked = _search(
            args.api_url, row["query"], args.top_k, args.hybrid, args.rerank
        )
        scores = score_query(row.get("relevant", []), ranked, args.top_k)
        per_query.append({"query": row["query"], "scores": scores})
        logger.info("Query %r -> %s", row["query"], scores)

    totals: dict[str, list[float]] = {}
    for entry in per_query:
        for name, value in entry["scores"].items():
            totals.setdefault(name, []).append(value)
    means = {name: sum(values) / len(values) for name, values in totals.items()}
    for name, value in means.items():
        print(f"{name}: {value:.4f}")

    if args.mlflow:
        from rag_platform.mlflow_tracking.tracker import log_metrics, track_run

        retrieval = (
            "hybrid+rerank"
            if args.hybrid and args.rerank
            else "hybrid"
            if args.hybrid
            else "dense"
        )
        params = {
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "n_queries": len(per_query),
            "retrieval": retrieval,
        }
        with track_run(run_name=f"retrieval-{retrieval}", params=params):
            log_metrics(means)
        logger.info("Retrieval evaluation (%s) logged to MLflow: %s", retrieval, means)


if __name__ == "__main__":
    main()
