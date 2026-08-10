"""API endpoint tests using FastAPI TestClient with overridden dependencies."""

from __future__ import annotations


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
