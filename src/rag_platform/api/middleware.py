"""HTTP request logging and metrics middleware."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware

from rag_platform.observability.metrics import metrics
from rag_platform.utils.logging import get_logger

logger = get_logger(__name__)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Log each request with its latency and record Prometheus-style metrics."""

    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_s = time.perf_counter() - start
        path = request.url.path
        metrics.record(request.method, path, response.status_code, duration_s)
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            path,
            response.status_code,
            duration_s * 1000,
        )
        return response
