"""FastAPI dependency providers for the embedder and vector store.

Singletons are created once per process and can be overridden in tests via
``app.dependency_overrides``.
"""

from __future__ import annotations

from rag_platform.config.settings import get_settings
from rag_platform.embeddings.provider import EmbeddingProvider, build_embedding_provider
from rag_platform.embeddings.sparse import HashingSparseEncoder
from rag_platform.vectorstore.base import VectorStore
from rag_platform.vectorstore.factory import build_vector_store

_embedder_singleton: EmbeddingProvider | None = None
_store_singleton: VectorStore | None = None
_sparse_singleton: HashingSparseEncoder | None = None


def get_embedder() -> EmbeddingProvider:
    """Return the process-wide embedding provider."""
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = build_embedding_provider(get_settings())
    return _embedder_singleton


def get_sparse_encoder() -> HashingSparseEncoder:
    """Return the process-wide sparse encoder for hybrid search."""
    global _sparse_singleton
    if _sparse_singleton is None:
        _sparse_singleton = HashingSparseEncoder(n_features=get_settings().sparse_dim)
    return _sparse_singleton


def get_vector_store() -> VectorStore:
    """Return the process-wide vector store (collection created on first use)."""
    global _store_singleton
    if _store_singleton is None:
        settings = get_settings()
        _store_singleton = build_vector_store(settings, get_embedder().dimension)
        _store_singleton.create_collection()
    return _store_singleton
