"""Shared pytest fixtures: hermetic embedder, store, and API client."""

from __future__ import annotations

import pytest

from rag_platform.api.main import create_app
from rag_platform.vectorstore.qdrant_store import QdrantStore
from tests.fakes import DummyEmbeddingProvider

_DIMENSION = 8
_COLLECTION = "test_docs"


@pytest.fixture()
def embedder() -> DummyEmbeddingProvider:
    return DummyEmbeddingProvider(dimension=_DIMENSION)


@pytest.fixture()
def store() -> QdrantStore:
    """Qdrant in-memory store; no server or network required."""
    qdrant = QdrantStore(path=":memory:", collection=_COLLECTION, vector_size=_DIMENSION)
    qdrant.create_collection()
    return qdrant


@pytest.fixture()
def client(store: QdrantStore, embedder: DummyEmbeddingProvider):
    from fastapi.testclient import TestClient

    from rag_platform.api.dependencies import (
        get_embedder,
        get_sparse_encoder,
        get_vector_store,
    )
    from rag_platform.embeddings.sparse import HashingSparseEncoder

    app = create_app()
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_sparse_encoder] = lambda: HashingSparseEncoder(n_features=64)
    with TestClient(app) as test_client:
        yield test_client
