"""MLflow experiment tracking helpers.

MLflow is imported lazily and only used inside ``track_run`` so serving and ingestion
paths that do not track experiments stay lightweight. Install with
``pip install -e ".[eval]"``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rag_platform.config.settings import get_settings
from rag_platform.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def track_run(run_name: str, params: dict[str, Any] | None = None) -> Iterator[None]:
    """Open an MLflow run, log ``params``, and yield inside the run context.

    Metrics and artifacts are logged inside the ``with`` block via ``log_metrics`` or
    ``mlflow.log_artifact``.
    """
    import mlflow

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        logger.info("MLflow run started: %s (%s)", run_name, run.info.run_id)
        yield
    logger.info("MLflow run finished: %s", run.info.run_id)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to the active MLflow run."""
    import mlflow

    mlflow.log_metrics(metrics, step=step)
