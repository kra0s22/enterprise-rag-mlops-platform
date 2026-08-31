"""Minimal Prometheus-style metrics collector (no external dependencies).

Tracks HTTP request counts by (method, route, status) and a latency histogram,
rendered in Prometheus text format by ``/metrics``. Kept dependency-free so the
serving image stays lean.
"""

from __future__ import annotations

import threading
from typing import Any

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class Metrics:
    """Thread-safe counters and a latency histogram for HTTP requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str, int], int] = {}
        self._durations: list[float] = []

    def record(self, method: str, route: str, status: int, duration_s: float) -> None:
        with self._lock:
            key = (method, route, status)
            self._counters[key] = self._counters.get(key, 0) + 1
            self._durations.append(duration_s)

    def render(self) -> str:
        """Render the collected metrics in Prometheus text exposition format."""
        with self._lock:
            lines = [
                "# HELP rag_http_requests_total HTTP requests by method, route and status.",
                "# TYPE rag_http_requests_total counter",
            ]
            for (method, route, status), count in sorted(self._counters.items()):
                lines.append(
                    f'rag_http_requests_total{{method="{method}",route="{route}",'
                    f'status="{status}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP rag_http_request_duration_seconds HTTP request latency.",
                    "# TYPE rag_http_request_duration_seconds histogram",
                ]
            )
            bucket_counts = [0] * len(_LATENCY_BUCKETS)
            for duration in self._durations:
                for index, bound in enumerate(_LATENCY_BUCKETS):
                    if duration <= bound:
                        bucket_counts[index] += 1
            for bound, count in zip(_LATENCY_BUCKETS, bucket_counts, strict=True):
                lines.append(
                    f'rag_http_request_duration_seconds_bucket{{le="{bound}"}} {count}'
                )
            lines.append(
                f'rag_http_request_duration_seconds_bucket{{le="+Inf"}} '
                f"{len(self._durations)}"
            )
            lines.append(f"rag_http_request_duration_seconds_count {len(self._durations)}")
            if self._durations:
                lines.append(
                    f"rag_http_request_duration_seconds_sum {sum(self._durations)}"
                )
        return "\n".join(lines) + "\n"


metrics: Any = Metrics()
