"""API security: optional API-key authentication and per-client rate limiting.

Auth is opt-in: when ``RAG_API_KEY`` is set, ``/v1/*`` routes require an
``X-API-Key`` header. The rate limiter enforces a sliding-window per-client cap
and returns 429 once it is exceeded.
"""

from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from rag_platform.config.settings import get_settings


def require_api_key(request: Request) -> None:
    """FastAPI dependency enforcing ``X-API-Key`` when ``RAG_API_KEY`` is set."""
    settings = get_settings()
    if not settings.api_key:
        return
    if request.headers.get("X-API-Key") != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP (returns 429 when exceeded)."""

    def __init__(self, app, limit_per_minute: int) -> None:
        super().__init__(app)
        self._limit = max(1, limit_per_minute)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            window = [stamp for stamp in self._hits.get(client, []) if now - stamp < 60.0]
            if len(window) >= self._limit:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded"},
                )
            window.append(now)
            self._hits[client] = window
        return await call_next(request)
