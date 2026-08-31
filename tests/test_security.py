"""Hermetic tests for API security (auth + rate limiting) and metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_platform.api.dependencies import (
    get_embedder,
    get_reranker,
    get_sparse_encoder,
    get_vector_store,
)
from rag_platform.api.main import create_app
from rag_platform.config.settings import get_settings
from rag_platform.embeddings.sparse import HashingSparseEncoder
from rag_platform.reranking.reranker import Reranker
from rag_platform.vectorstore.qdrant_store import QdrantStore
from tests.fakes import DummyEmbeddingProvider


def _app_with_fakes() -> TestClient:
    embedder = DummyEmbeddingProvider(dimension=8)
    store = QdrantStore(path=":memory:", collection="test_security", vector_size=8)
    store.create_collection()

    class _FakeReranker(Reranker):
        def score(self, query, documents):
            return [1.0 if query in document else 0.0 for document in documents]

    app = create_app()
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_sparse_encoder] = lambda: HashingSparseEncoder(n_features=64)
    app.dependency_overrides[get_reranker] = lambda: _FakeReranker()
    return TestClient(app)


def test_health_is_open_without_api_key() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200


def test_api_key_required_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "api_key", "secret")
    with _app_with_fakes() as client:
        no_key = client.post("/v1/search", json={"query": "q"})
        assert no_key.status_code == 401

        with_key = client.post(
            "/v1/search", json={"query": "q"}, headers={"X-API-Key": "secret"}
        )
        assert with_key.status_code == 200


def test_rate_limit_returns_429(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 429


def test_metrics_endpoint_renders_prometheus_format() -> None:
    with TestClient(create_app()) as client:
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "rag_http_requests_total" in body
        assert "rag_http_request_duration_seconds_bucket" in body
