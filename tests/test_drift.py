"""Hermetic tests for knowledge-base drift monitoring (no network or models)."""

from __future__ import annotations

import numpy as np
import pytest

from rag_platform.observability.drift import (
    DEFAULT_DRIFT_THRESHOLDS,
    CorpusSnapshot,
    collect_snapshot,
    cosine_similarity,
    detect_drift,
    diff_sources,
    drift_metrics,
    embedding_snapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)


def _snapshot(vectors, payloads=None) -> CorpusSnapshot:
    return embedding_snapshot(vectors, payloads or [{} for _ in vectors])


def test_embedding_snapshot_computes_centroid_and_norm() -> None:
    vectors = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    snapshot = _snapshot(vectors)
    assert snapshot.chunk_count == 2
    assert np.allclose(snapshot.centroid, [0.5, 0.5])
    assert snapshot.mean_norm == 1.0  # both unit vectors


def test_embedding_snapshot_empty() -> None:
    snapshot = embedding_snapshot([], [])
    assert snapshot.chunk_count == 0
    assert snapshot.centroid.size == 0
    assert snapshot.mean_norm == 0.0


def test_embedding_snapshot_extracts_sources_and_freshness() -> None:
    vectors = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    payloads = [
        {"source": "a.md", "ingested_at": "2026-08-31T10:00:00+00:00"},
        {"metadata": {"source": "b.md"}, "ingested_at": "2026-08-31T11:00:00+00:00"},
    ]
    snapshot = embedding_snapshot(vectors, payloads)
    assert snapshot.sources == {"a.md", "b.md"}
    assert snapshot.last_ingested_at == "2026-08-31T11:00:00+00:00"


def test_embedding_snapshot_skips_missing_freshness() -> None:
    vectors = [np.array([1.0, 0.0], dtype=np.float32)]
    snapshot = _snapshot(vectors, [{"source": "a.md"}])
    assert snapshot.last_ingested_at is None


def test_cosine_similarity_identical() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector() -> None:
    zero = np.zeros(3, dtype=np.float32)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(zero, vector) == 0.0


def test_drift_metrics_identical_snapshot() -> None:
    baseline = _snapshot([np.array([1.0, 0.0], dtype=np.float32)])
    metrics = drift_metrics(baseline, baseline)
    assert metrics["centroid_drift"] == pytest.approx(0.0)
    assert metrics["count_delta"] == 0.0
    assert metrics["count_delta_ratio"] == 0.0


def test_drift_metrics_count_growth() -> None:
    baseline = _snapshot([np.array([1.0, 0.0], dtype=np.float32)])
    current = _snapshot([np.array([1.0, 0.0], dtype=np.float32)] * 2)
    metrics = drift_metrics(baseline, current)
    assert metrics["count_delta"] == 1.0
    assert metrics["count_delta_ratio"] == 1.0  # doubled corpus


def test_drift_metrics_centroid_shift() -> None:
    baseline = _snapshot([np.array([1.0, 0.0], dtype=np.float32)])
    current = _snapshot([np.array([0.0, 1.0], dtype=np.float32)])
    metrics = drift_metrics(baseline, current)
    assert metrics["centroid_cosine"] == pytest.approx(0.0)
    assert metrics["centroid_drift"] == pytest.approx(1.0)


def test_detect_drift_centroid_threshold() -> None:
    assert detect_drift({"centroid_drift": 0.01, "count_delta_ratio": 0.0}) is False
    assert detect_drift({"centroid_drift": 0.10, "count_delta_ratio": 0.0}) is True


def test_detect_drift_count_ratio_threshold() -> None:
    assert detect_drift({"centroid_drift": 0.0, "count_delta_ratio": 0.10}) is False
    assert detect_drift({"centroid_drift": 0.0, "count_delta_ratio": 0.50}) is True


def test_detect_drift_custom_thresholds() -> None:
    tighter = {"centroid_drift": 0.01, "count_delta_ratio": 0.05}
    assert detect_drift({"centroid_drift": 0.02, "count_delta_ratio": 0.0}, tighter) is True
    assert detect_drift({"centroid_drift": 0.0, "count_delta_ratio": 0.10}, tighter) is True


def test_diff_sources_added_and_removed() -> None:
    baseline = CorpusSnapshot(1, np.array([1.0], dtype=np.float32), 1.0, {"a", "b"})
    current = CorpusSnapshot(1, np.array([1.0], dtype=np.float32), 1.0, {"b", "c"})
    result = diff_sources(baseline, current)
    assert result["added"] == {"c"}
    assert result["removed"] == {"a"}


class _FakeStore:
    def __init__(self, points) -> None:
        self._points = points

    def iter_points(self):
        return self._points


def test_collect_snapshot_uses_iter_points() -> None:
    store = _FakeStore(
        [
            ("id-1", np.array([1.0, 0.0], dtype=np.float32), {"source": "a.md"}),
            ("id-2", np.array([0.0, 1.0], dtype=np.float32), {"source": "b.md"}),
        ]
    )
    snapshot = collect_snapshot(store)
    assert snapshot.chunk_count == 2
    assert snapshot.sources == {"a.md", "b.md"}
    assert np.allclose(snapshot.centroid, [0.5, 0.5])


def test_collect_snapshot_empty_store() -> None:
    snapshot = collect_snapshot(_FakeStore([]))
    assert snapshot.chunk_count == 0


def test_snapshot_json_roundtrip() -> None:
    snapshot = CorpusSnapshot(
        chunk_count=3,
        centroid=np.array([0.2, 0.8], dtype=np.float32),
        mean_norm=0.91,
        sources={"a.md", "b.md"},
        last_ingested_at="2026-08-31T11:00:00+00:00",
    )
    restored = snapshot_from_dict(snapshot_to_dict(snapshot))
    assert restored.chunk_count == snapshot.chunk_count
    assert restored.mean_norm == pytest.approx(snapshot.mean_norm)
    assert restored.sources == snapshot.sources
    assert restored.last_ingested_at == snapshot.last_ingested_at
    assert np.allclose(restored.centroid, snapshot.centroid)


def test_default_thresholds_are_sane() -> None:
    assert DEFAULT_DRIFT_THRESHOLDS["centroid_drift"] > 0
    assert DEFAULT_DRIFT_THRESHOLDS["count_delta_ratio"] > 0
