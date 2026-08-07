"""Vector store factory that builds the configured backend."""

from __future__ import annotations

from typing import Any

from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import VectorStore
from rag_platform.vectorstore.milvus_store import MilvusStore
from rag_platform.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


def build_vector_store(settings: Any, dimension: int) -> VectorStore:
    """Instantiate the vector store selected by ``settings.vector_store_provider``."""
    provider = settings.vector_store_provider.lower()
    if provider == "qdrant":
        return QdrantStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            vector_size=dimension,
            api_key=settings.qdrant_api_key,
        )
    if provider == "milvus":
        return MilvusStore(
            uri=settings.milvus_uri,
            collection=settings.milvus_collection,
            vector_size=dimension,
            token=settings.milvus_token,
        )
    raise ValueError(f"Unsupported vector store provider: {settings.vector_store_provider}")
