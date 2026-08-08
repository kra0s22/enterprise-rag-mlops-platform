"""Ragas-based RAG evaluation helpers.

The heavy ``ragas`` dependency (and its langchain/datasets graph) is imported lazily so
core serving paths never load it. Install with ``pip install -e ".[eval]"``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from rag_platform.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def evaluate_rag(
    dataset: list[dict[str, Any]], metrics: list[str] | None = None
) -> dict[str, float]:
    """Evaluate a RAG dataset with Ragas and return metric scores.

    Args:
        dataset: List of samples with keys ``question``, ``answer``, ``contexts``
            (list of retrieved chunks) and optional ``ground_truth``.
        metrics: Metric names to compute; defaults to the four core metrics.

    Returns:
        Mapping of metric name to score in [0, 1].
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    selected = metrics or DEFAULT_METRICS
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row.get("answer", ""),
            retrieved_contexts=row.get("contexts", []),
            reference=row.get("ground_truth"),
        )
        for row in dataset
    ]
    chosen = [metric_map[name] for name in selected if name in metric_map]
    if not chosen:
        raise ValueError(f"No valid metrics requested; supported: {list(metric_map)}")

    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=chosen)
    scores = {name: float(getattr(result, name)) for name in selected if name in metric_map}
    logger.info("Ragas evaluation completed: %s", scores)
    return scores


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on a JSONL dataset.")
    parser.add_argument("--dataset", required=True, type=str, help="Path to a JSONL dataset file")
    args = parser.parse_args()

    dataset: list[dict[str, Any]] = []
    with open(args.dataset, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    scores = evaluate_rag(dataset)
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    _main()
