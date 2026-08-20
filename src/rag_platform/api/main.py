"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_platform.api.routes import ingest, rag, search
from rag_platform.config.settings import get_settings
from rag_platform.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(get_settings().log_level)
    logger.info("RAG Platform API starting")
    yield
    logger.info("RAG Platform API shutting down")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="RAG Platform API",
        version="0.1.0",
        description="Enterprise RAG: document ingestion and retrieval endpoints.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ingest.router)
    app.include_router(search.router)
    app.include_router(rag.router)

    @app.get("/health", tags=["system"], summary="Liveness probe")
    def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name, "environment": settings.environment}

    return app


app = create_app()


def run() -> None:
    """Console entry point: run the uvicorn server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag_platform.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
