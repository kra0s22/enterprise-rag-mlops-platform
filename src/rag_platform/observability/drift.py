"""Knowledge-base drift monitoring (pure, backend-agnostic).

Corpus embeddings shift as documents are added, removed or rewritten. This module
captures an embedding snapshot of the whole collection and compares it against a
baseline to flag drift: centroid shift, chunk-count delta and source churn.

The functions here are pure (numpy in, plain values out) so the logic is fully
covered by the hermetic test suite; the ``run_drift_monitor`` CLI wires them to a
live vector store and MLflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Maximum allowed centroid drift (1 - cosine similarity) and relative chunk-count
# change before a corpus is flagged as drifted.
DEFAULT_DRIFT_THRESHOLDS: dict[str, float] = {
    "centroid_drift": 0.05,
    "count_delta_ratio": 0.20,
}


@dataclass(slots=True)
class CorpusSnapshot:
    """Statistics describing the embedding distribution of a collection."""

    chunk_count: int
    centroid: np.ndarray
    mean_norm: float
    sources: set[str] = field(default_factory=set)
    last_ingested_at: str | None = None


def _extract_source(payload: dict[str, Any]) -> str | None:
    source = payload.get("source")
    if source is None:
        source = payload.get("metadata", {}).get("source")
    return str(source) if source else None


def embedding_snapshot(
    vectors: Sequence[np.ndarray],
    payloads: Sequence[dict[str, Any]],
) -> CorpusSnapshot:
    """Summarise a collection of dense vectors and their payloads.

    The centroid is the mean embedding vector; on a clean corpus it sits close to
    the baseline centroid, while new topics pull it in a new direction. ``sources``
    and ``last_ingested_at`` support content-freshness reporting.
    """
    if not vectors:
        return CorpusSnapshot(
            chunk_count=0, centroid=np.zeros(0, dtype=np.float32), mean_norm=0.0
        )
    matrix = np.asarray(vectors, dtype=np.float32)
    centroid = matrix.mean(axis=0)
    mean_norm = float(np.linalg.norm(matrix, axis=1).mean())
    sources = {source for payload in payloads if (source := _extract_source(payload))}
    ingested = [
        str(payload["ingested_at"])
        for payload in payloads
        if isinstance(payload.get("ingested_at"), str) and payload["ingested_at"]
    ]
    return CorpusSnapshot(
        chunk_count=len(vectors),
        centroid=centroid,
        mean_norm=mean_norm,
        sources=sources,
        last_ingested_at=max(ingested) if ingested else None,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, or 0.0 when either is zero."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def drift_metrics(baseline: CorpusSnapshot, current: CorpusSnapshot) -> dict[str, float]:
    """Compare a baseline against the current snapshot and return drift metrics."""
    centroid_cosine = cosine_similarity(baseline.centroid, current.centroid)
    if baseline.chunk_count > 0:
        count_ratio = current.chunk_count / baseline.chunk_count
    else:
        count_ratio = float("inf") if current.chunk_count > 0 else 1.0
    return {
        "centroid_cosine": centroid_cosine,
        "centroid_drift": 1.0 - centroid_cosine,
        "count_delta": float(current.chunk_count - baseline.chunk_count),
        "count_delta_ratio": count_ratio - 1.0,
        "norm_delta": current.mean_norm - baseline.mean_norm,
    }


def detect_drift(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> bool:
    """Return True when any drift metric exceeds its configured threshold."""
    thresholds = thresholds or DEFAULT_DRIFT_THRESHOLDS
    if metrics.get("centroid_drift", 0.0) > thresholds["centroid_drift"]:
        return True
    if abs(metrics.get("count_delta_ratio", 0.0)) > thresholds["count_delta_ratio"]:
        return True
    return False


def diff_sources(
    baseline: CorpusSnapshot, current: CorpusSnapshot
) -> dict[str, set[str]]:
    """Return the sources added to and removed from the corpus since baseline."""
    return {
        "added": current.sources - baseline.sources,
        "removed": baseline.sources - current.sources,
    }


def collect_snapshot(store: Any) -> CorpusSnapshot:
    """Snapshot the whole collection through a VectorStore's ``iter_points``."""
    points = store.iter_points()
    vectors = [point[1] for point in points]
    payloads = [point[2] for point in points]
    return embedding_snapshot(vectors, payloads)


def snapshot_to_dict(snapshot: CorpusSnapshot) -> dict[str, Any]:
    """Serialise a snapshot to JSON-compatible values for baseline storage."""
    return {
        "chunk_count": snapshot.chunk_count,
        "centroid": snapshot.centroid.tolist(),
        "mean_norm": snapshot.mean_norm,
        "sources": sorted(snapshot.sources),
        "last_ingested_at": snapshot.last_ingested_at,
    }


def snapshot_from_dict(data: dict[str, Any]) -> CorpusSnapshot:
    """Rebuild a snapshot from ``snapshot_to_dict`` output."""
    return CorpusSnapshot(
        chunk_count=int(data["chunk_count"]),
        centroid=np.asarray(data["centroid"], dtype=np.float32),
        mean_norm=float(data["mean_norm"]),
        sources=set(data.get("sources", [])),
        last_ingested_at=data.get("last_ingested_at"),
    )
