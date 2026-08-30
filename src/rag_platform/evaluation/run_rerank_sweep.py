"""Rerank candidate-pool sweep over the chunk-level retrieval eval.

Varies the number of candidates retrieved before cross-encoder reranking
(``rerank_top_k``) and scores each pool size with MRR@k / hit-rate@k / nDCG@k,
logging the runs to MLflow. Larger pools can raise recall at the cost of rerank
latency, so the sweep exposes the quality/latency trade-off.

Usage:
    python -m rag_platform.evaluation.run_rerank_sweep --pools 5,10,20,30,40 --mlflow
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from rag_platform.config.settings import get_settings
from rag_platform.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

METRIC_NAMES = ("mrr_at_k", "hit_rate_at_k", "ndcg_at_k")


def _run(command: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(
        command, env=env, capture_output=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        logger.error("Command failed (%s): %s", proc.returncode, " ".join(command))
        logger.error("stdout tail:\n%s", proc.stdout[-2000:])
        logger.error("stderr tail:\n%s", proc.stderr[-2000:])
        proc.check_returncode()
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the rerank candidate pool on retrieval metrics."
    )
    parser.add_argument("--pools", default="5,10,20,30,40", help="Comma-separated pool sizes")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="RAG API base URL")
    parser.add_argument("--top-k", default=5, type=int, help="Ranked chunks returned")
    parser.add_argument(
        "--dataset", default=Path("data/retrieval_set.jsonl"), type=Path, help="Retrieval dataset"
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--chunk-mode", default="window")
    parser.add_argument("--mlflow", action="store_true", help="Log each pool size to MLflow")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    pools = [int(value) for value in args.pools.split(",")]
    results: list[dict[str, float]] = []

    for pool in pools:
        command = [
            sys.executable,
            "-m",
            "rag_platform.evaluation.run_retrieval_eval",
            "--dataset",
            str(args.dataset),
            "--api-url",
            args.api_url,
            "--top-k",
            str(args.top_k),
            "--chunk-size",
            str(args.chunk_size),
            "--chunk-overlap",
            str(args.chunk_overlap),
            "--chunk-mode",
            args.chunk_mode,
            "--hybrid",
            "--rerank",
            "--rerank-top-k",
            str(pool),
        ]
        if args.mlflow:
            command.append("--mlflow")
        stdout = _run(command, os.environ)

        metrics: dict[str, float] = {}
        for line in stdout.splitlines():
            if ":" in line:
                name, value = line.split(":", 1)
                if name.strip() in METRIC_NAMES:
                    metrics[name.strip()] = float(value.strip())
        row = {"rerank_top_k": float(pool), **metrics}
        results.append(row)
        logger.info("pool=%d -> %s", pool, metrics)

    print("\n=== RERANK POOL SUMMARY (best nDCG first) ===")
    for row in sorted(results, key=lambda r: -r.get("ndcg_at_k", 0.0)):
        print(
            f"pool={int(row['rerank_top_k']):3d}  "
            f"mrr={row.get('mrr_at_k', 0.0):.4f} hit={row.get('hit_rate_at_k', 0.0):.4f} "
            f"ndcg={row.get('ndcg_at_k', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
