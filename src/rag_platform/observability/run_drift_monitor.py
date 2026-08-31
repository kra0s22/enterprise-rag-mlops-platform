"""CLI to capture and monitor knowledge-base drift.

Usage::

    # Record the reference snapshot after a (re)ingest.
    python -m rag_platform.observability.run_drift_monitor --baseline [--mlflow]

    # Periodically compare the live collection against the baseline.
    python -m rag_platform.observability.run_drift_monitor [--mlflow]

The default run prints a human-readable report, logs drift metrics to MLflow with
``--mlflow``, and exits 1 when drift exceeds the configured thresholds so the job
can be wired into a scheduler (cron, CI, orchestrator).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_platform.config.settings import get_settings
from rag_platform.observability.drift import (
    DEFAULT_DRIFT_THRESHOLDS,
    collect_snapshot,
    detect_drift,
    diff_sources,
    drift_metrics,
    snapshot_from_dict,
    snapshot_to_dict,
)
from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.factory import build_vector_store

logger = get_logger(__name__)

DEFAULT_BASELINE_PATH = Path("data/drift_baseline.json")


def _build_store() -> object:
    settings = get_settings()
    return build_vector_store(settings, settings.embedding_dimension)


def _log_mlflow(kind: str, snapshot: object, metrics: dict[str, float] | None) -> None:
    """Log a drift run to MLflow (heavy dependency, imported lazily)."""
    import mlflow

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"drift-{kind}"):
        mlflow.log_param("kind", kind)
        mlflow.log_param("chunk_count", snapshot.chunk_count)
        mlflow.log_param("mean_norm", round(snapshot.mean_norm, 6))
        mlflow.log_param("sources", ",".join(sorted(snapshot.sources)))
        mlflow.log_param("last_ingested_at", snapshot.last_ingested_at or "")
        if metrics:
            for name, value in metrics.items():
                mlflow.log_metric(name, round(float(value), 6))


def _write_baseline(snapshot: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot_to_dict(snapshot), indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture or monitor knowledge-base drift."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Record the current collection as the reference snapshot and exit.",
    )
    parser.add_argument("--baseline-path", type=str, default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument(
        "--mlflow", action="store_true", help="Log the snapshot/metrics to MLflow."
    )
    args = parser.parse_args(argv)

    store = _build_store()
    current = collect_snapshot(store)
    baseline_path = Path(args.baseline_path)

    if args.baseline:
        _write_baseline(current, baseline_path)
        if args.mlflow:
            _log_mlflow("baseline", current, None)
        logger.info("Baseline saved: %s (chunks=%d)", baseline_path, current.chunk_count)
        print(f"baseline chunks={current.chunk_count} sources={len(current.sources)}")
        return 0

    if not baseline_path.exists():
        logger.error(
            "No baseline at %s; run with --baseline after (re)ingesting first",
            baseline_path,
        )
        print("error: no baseline found; run with --baseline first", file=sys.stderr)
        return 2

    baseline = snapshot_from_dict(json.loads(baseline_path.read_text(encoding="utf-8")))
    metrics = drift_metrics(baseline, current)
    drifted = detect_drift(metrics, DEFAULT_DRIFT_THRESHOLDS)
    sources_diff = diff_sources(baseline, current)

    print(f"chunks:      {baseline.chunk_count} -> {current.chunk_count}")
    print(f"centroid:    cosine={metrics['centroid_cosine']:.4f} "
          f"drift={metrics['centroid_drift']:.4f}")
    print(f"count ratio: {metrics['count_delta_ratio']:+.2%}")
    print(f"norm delta:  {metrics['norm_delta']:+.4f}")
    if sources_diff["added"]:
        print(f"added:       {', '.join(sorted(sources_diff['added']))}")
    if sources_diff["removed"]:
        print(f"removed:     {', '.join(sorted(sources_diff['removed']))}")
    print(f"drift:       {'DETECTED' if drifted else 'ok'}")

    if args.mlflow:
        _log_mlflow("monitor", current, metrics)
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
