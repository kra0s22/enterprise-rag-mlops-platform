"""Chunk-size and overlap ablation over the chunk-level retrieval eval.

Re-ingests the sample corpus for every ``(chunk_size, chunk_overlap)``
combination, scores the ranking with MRR@k / hit-rate@k / nDCG@k and logs each
configuration to MLflow, so the chunking that maximizes retrieval quality can be
chosen empirically instead of by guesswork.

Usage:
    python -m rag_platform.evaluation.run_ablation \\
        --chunk-sizes 200,300,400,512 --overlaps 32,64,128 --mlflow
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import httpx

from rag_platform.config.settings import get_settings
from rag_platform.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

METRIC_NAMES = ("mrr_at_k", "hit_rate_at_k", "ndcg_at_k")


def _delete_collection(qdrant_url: str, collection: str) -> None:
    httpx.delete(f"{qdrant_url}/collections/{collection}", timeout=30).raise_for_status()


def _run(command: list[str], env: dict[str, str]) -> str:
    # MLflow prints a UTF-8 emoji banner; capture as UTF-8 so Windows' default
    # cp1252 codec does not raise UnicodeDecodeError on the child output.
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
        description="Ablate chunk_size and chunk_overlap on retrieval metrics."
    )
    parser.add_argument("--chunk-sizes", default="200,300,400,512", help="Comma-separated sizes")
    parser.add_argument("--overlaps", default="32,64,128", help="Comma-separated overlaps")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="RAG API base URL")
    parser.add_argument("--top-k", default=5, type=int, help="Ranked chunks considered")
    parser.add_argument(
        "--dataset", default=Path("data/retrieval_set.jsonl"), type=Path, help="Retrieval dataset"
    )
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid dense+sparse retrieval")
    parser.add_argument(
        "--rerank", action="store_true", help="Rerank candidates with the cross-encoder"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="Log each configuration to MLflow"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    chunk_sizes = [int(value) for value in args.chunk_sizes.split(",")]
    overlaps = [int(value) for value in args.overlaps.split(",")]

    results: list[dict[str, float]] = []
    for chunk_size in chunk_sizes:
        for chunk_overlap in overlaps:
            _delete_collection(settings.qdrant_url, settings.qdrant_collection)

            env = {**os.environ, "RAG_CHUNK_SIZE": str(chunk_size),
                   "RAG_CHUNK_OVERLAP": str(chunk_overlap)}
            _run(
                [sys.executable, "-m", "rag_platform.ingestion.cli", "./data/sample"],
                env,
            )

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
                str(chunk_size),
                "--chunk-overlap",
                str(chunk_overlap),
            ]
            if args.hybrid:
                command.append("--hybrid")
            if args.rerank:
                command.append("--rerank")
            if args.mlflow:
                command.append("--mlflow")
            stdout = _run(command, env)

            metrics = {}
            for line in stdout.splitlines():
                if ":" in line:
                    name, value = line.split(":", 1)
                    if name.strip() in METRIC_NAMES:
                        metrics[name.strip()] = float(value.strip())
            row = {
                "chunk_size": float(chunk_size),
                "chunk_overlap": float(chunk_overlap),
                **metrics,
            }
            results.append(row)
            logger.info("cs=%d ov=%d -> %s", chunk_size, chunk_overlap, metrics)

    print("\n=== ABLATION SUMMARY (best nDCG first) ===")
    for row in sorted(results, key=lambda r: -r.get("ndcg_at_k", 0.0)):
        print(
            f"cs={int(row['chunk_size']):4d} ov={int(row['chunk_overlap']):3d}  "
            f"mrr={row.get('mrr_at_k', 0.0):.4f} hit={row.get('hit_rate_at_k', 0.0):.4f} "
            f"ndcg={row.get('ndcg_at_k', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
