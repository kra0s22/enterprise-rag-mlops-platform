"""End-to-end evaluation runner for the /v1/rag endpoint.

Fetches a grounded answer for every question in the evaluation dataset from the
live API, persists the collected samples, and scores the responses with Ragas
using a self-hosted LLM (Ollama) through its OpenAI-compatible endpoint.

Usage:
    python -m rag_platform.evaluation.run_evaluation \\
        --dataset data/eval_set.jsonl --api-url http://127.0.0.1:8000 --top-k 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import httpx

from rag_platform.config.settings import get_settings
from rag_platform.evaluation.ragas_eval import evaluate_rag
from rag_platform.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def collect_answers(
    dataset: list[dict[str, Any]],
    api_url: str,
    top_k: int,
    timeout: float = 300.0,
) -> list[dict[str, Any]]:
    """Query /v1/rag for each question and build Ragas samples with the answers.

    Each returned sample keeps ``question``, ``answer``, the retrieved
    ``contexts`` and the optional ``ground_truth`` for reference-based metrics.
    """
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=api_url, timeout=timeout) as client:
        for row in dataset:
            response = client.post("/v1/rag", json={"query": row["question"], "top_k": top_k})
            response.raise_for_status()
            body = response.json()
            results.append(
                {
                    "question": row["question"],
                    "answer": body["answer"],
                    "contexts": [source["chunk_text"] for source in body["sources"]],
                    "ground_truth": row.get("ground_truth"),
                }
            )
            logger.info(
                "Collected answer for %r (%d sources)", row["question"], len(body["sources"])
            )
    return results


def build_local_llm() -> Any:
    """Build a Ragas LLM wrapper backed by Ollama's OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    chat = ChatOpenAI(
        model=settings.llm_model,
        base_url=f"{settings.ollama_url}/v1",
        api_key="ollama",  # placeholder; Ollama does not enforce authentication
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return LangchainLLMWrapper(chat)


def build_local_embeddings() -> Any:
    """Build Ragas embeddings from the same sentence-transformers model as retrieval."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ragas evaluation against the live RAG API."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL eval dataset")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="RAG API base URL")
    parser.add_argument("--top-k", default=3, type=int, help="Retrieved chunks per query")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Ragas metrics to compute",
    )
    parser.add_argument(
        "--out", default=Path("data/eval_results.jsonl"), type=Path, help="Output JSONL path"
    )
    parser.add_argument(
        "--mlflow", action="store_true", help="Log experiment parameters and metrics to MLflow"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    # Windows consoles default to cp1252, which MLflow's run-URL banner (emoji)
    # cannot encode; force UTF-8 so ending a run never crashes on stdout.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = collect_answers(rows, args.api_url, args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples),
        encoding="utf-8",
    )
    logger.info("Wrote %d collected samples to %s", len(samples), args.out)

    scores = evaluate_rag(
        samples,
        metrics=args.metrics,
        llm=build_local_llm(),
        embeddings=build_local_embeddings(),
    )
    for name, value in scores.items():
        label = "N/A" if math.isnan(value) else f"{value:.4f}"
        print(f"{name}: {label}")

    if args.mlflow:
        from rag_platform.mlflow_tracking.tracker import log_metrics, track_run

        params = {
            "llm_model": settings.llm_model,
            "llm_temperature": settings.llm_temperature,
            "embedding_model": settings.embedding_model,
            "top_k": args.top_k,
            "dataset": str(args.dataset),
            "n_samples": len(samples),
        }
        metrics = {name: value for name, value in scores.items() if not math.isnan(value)}
        with track_run(run_name=f"rag-eval-{settings.llm_model}", params=params):
            log_metrics(metrics)
        logger.info("Evaluation logged to MLflow with metrics: %s", metrics)


if __name__ == "__main__":
    main()
