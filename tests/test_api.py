"""API endpoint tests using FastAPI TestClient with overridden dependencies."""

from __future__ import annotations

import uuid

from rag_platform.embeddings.sparse import HashingSparseEncoder
from rag_platform.vectorstore.base import VectorStore


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app" in body


def test_ingest_document(client, store) -> None:
    response = client.post(
        "/v1/ingest",
        json={
            "text": "enterprise RAG systems combine retrieval with generation",
            "metadata": {"source": "test"},
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["num_chunks"] == 1
    assert len(body["stored_ids"]) == 1
    assert body["document_id"]
    assert store.count() == 1


def test_ingest_requires_text(client) -> None:
    response = client.post("/v1/ingest", json={"text": ""})
    assert response.status_code == 422


def test_search_roundtrip(client) -> None:
    client.post("/v1/ingest", json={"text": "the cat sat on the mat"})
    response = client.post("/v1/search", json={"query": "the cat sat on the mat", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "the cat sat on the mat"
    assert len(body["hits"]) >= 1
    assert body["hits"][0]["chunk_text"] == "the cat sat on the mat"
    assert body["hits"][0]["score"] > 0.9


def test_search_validates_top_k(client) -> None:
    response = client.post("/v1/search", json={"query": "hello", "top_k": 0})
    assert response.status_code == 422


def test_search_hybrid_returns_hits(client, store, embedder) -> None:
    texts = ["the cat sat on the mat", "quantum physics is tiny"]
    encoder = HashingSparseEncoder(n_features=64)
    store.upsert(
        [str(uuid.uuid4()), str(uuid.uuid4())],
        embedder.embed_documents(texts),
        [{"chunk_text": t} for t in texts],
        sparse_vectors=encoder.encode_batch(texts),
    )

    response = client.post("/v1/search", json={"query": "cat mat", "top_k": 2, "hybrid": True})

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits and hits[0]["chunk_text"] == "the cat sat on the mat"


def test_search_hybrid_unsupported_backend_returns_400() -> None:
    from fastapi.testclient import TestClient

    from rag_platform.api.dependencies import get_vector_store
    from rag_platform.api.main import create_app

    class _DenseOnly(VectorStore):
        def create_collection(self) -> None:
            return None

        def upsert(self, ids, vectors, payloads, sparse_vectors=None) -> None:
            return None

        def search(self, query_vector, top_k=5, filters=None):
            return []

        def delete(self, ids) -> None:
            return None

        def count(self) -> int:
            return 0

    app = create_app()
    app.dependency_overrides[get_vector_store] = lambda: _DenseOnly()
    with TestClient(app) as test_client:
        response = test_client.post("/v1/search", json={"query": "q", "hybrid": True})
    assert response.status_code == 400


def test_search_rerank_reorders_by_cross_encoder(client, store, embedder) -> None:
    texts = ["quantum physics is tiny", "the cat sat on the mat"]
    store.upsert(
        [str(uuid.uuid4()), str(uuid.uuid4())],
        embedder.embed_documents(texts),
        [{"chunk_text": t} for t in texts],
    )

    response = client.post("/v1/search", json={"query": "cat", "top_k": 1, "rerank": True})

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits and hits[0]["chunk_text"] == "the cat sat on the mat"
