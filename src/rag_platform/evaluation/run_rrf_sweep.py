"""RRF constant sweep over the chunk-level retrieval eval.

Qdrant's native RRF uses a fixed constant and the client exposes no knob, so this
runner fuses the dense and sparse rankings client-side with a configurable RRF
``k`` and scores each setting with MRR@k / nDCG@k, logging the runs to MLflow.

Usage:
    python -m rag_platform.evaluation.run_rrf_sweep --ks 30,60,90,120 --mlflow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rag_platform.api.dependencies import get_embedder, get_sparse_encoder
from rag_platform.config.settings import get_settings
from rag_platform.evaluation.retrieval_metrics import (
    ChunkKey,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
)
from rag_platform.evaluation.run_retrieval_eval import resolve_relevant
from rag_platform.utils.logging import configure_logging, get_logger
from rag_platform.vectorstore.fusion import keys_from_hits, rrf_fuse
from rag_platform.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


def _to_chunk_key(key: str) -> ChunkKey:
    source, index = key.rsplit("#", 1)
    return source, int(index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the RRF constant on chunk-level retrieval metrics."
    )
    parser.add_argument("--ks", default="30,60,90,120", help="Comma-separated RRF constants")
    parser.add_argument("--dataset", default=Path("data/retrieval_set.jsonl"), type=Path)
    parser.add_argument("--top-k", default=5, type=int, help="Ranked chunks considered")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument(
        "--mlflow", action="store_true", help="Log each RRF setting to MLflow"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    chunk_size = args.chunk_size if args.chunk_size else settings.chunk_size
    chunk_overlap = (
        args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap
    )
    ks = [int(value) for value in args.ks.split(",")]

    embedder = get_embedder()
    sparse_encoder = get_sparse_encoder()
    store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    per_query: list[dict[str, Any]] = []
    for row in rows:
        relevant = resolve_relevant(row.get("relevant", []), chunk_size, chunk_overlap)
        query = row["query"]
        dense_keys = keys_from_hits(store.search(embedder.embed_query(query), top_k=args.top_k))
        sparse_keys = keys_from_hits(
            store.search_sparse(sparse_encoder.encode(query), top_k=args.top_k)
        )
        per_query.append(
            {"query": query, "relevant": relevant, "dense": dense_keys, "sparse": sparse_keys}
        )
        logger.info("Resolved %d relevant chunks for %r", len(relevant), query)

    for k in ks:
        per_query_scores = []
        for entry in per_query:
            fused = rrf_fuse([entry["dense"], entry["sparse"]], k=k)
            ranked = [_to_chunk_key(key) for key in fused][: args.top_k]
            per_query_scores.append(
                {
                    "mrr_at_k": mrr_at_k(entry["relevant"], ranked, args.top_k),
                    "hit_rate_at_k": hit_rate_at_k(entry["relevant"], ranked, args.top_k),
                    "ndcg_at_k": ndcg_at_k(entry["relevant"], ranked, args.top_k),
                }
            )
        means = {
            name: sum(score[name] for score in per_query_scores) / len(per_query_scores)
            for name in ("mrr_at_k", "hit_rate_at_k", "ndcg_at_k")
        }
        print(f"rrf_k={k}: {means}")

        if args.mlflow:
            from rag_platform.mlflow_tracking.tracker import log_metrics, track_run

            params = {
                "rrf_k": k,
                "dataset": str(args.dataset),
                "top_k": args.top_k,
                "n_queries": len(per_query),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            with track_run(run_name=f"rrf-sweep-k{k}", params=params):
                log_metrics(means)
            logger.info("RRF sweep (k=%d) logged to MLflow: %s", k, means)


if __name__ == "__main__":
    main()
