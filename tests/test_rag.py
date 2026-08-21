"""Tests for the /v1/rag grounded generation endpoint."""

from __future__ import annotations

import pytest

from rag_platform.api.routes import rag as rag_route


class FakeGenerator:
    """Stand-in for the Ollama client that records calls and returns a fixed answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def generate(self, query: str, contexts: list[str]) -> str:
        self.calls.append((query, contexts))
        return "A grounded answer based on the provided context."


@pytest.fixture()
def fake_generator(monkeypatch) -> FakeGenerator:
    generator = FakeGenerator()
    monkeypatch.setattr(rag_route, "build_generation_client", lambda settings: generator)
    return generator


def _ingest(client, text: str) -> None:
    response = client.post("/v1/ingest", json={"text": text})
    assert response.status_code == 201


def test_rag_generates_grounded_answer(client, fake_generator) -> None:
    _ingest(client, "Qdrant stores embeddings as high-dimensional vectors for semantic search.")
    response = client.post(
        "/v1/rag", json={"query": "how does qdrant store embeddings", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A grounded answer based on the provided context."
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["chunk_text"]

    query, contexts = fake_generator.calls[0]
    assert query == "how does qdrant store embeddings"
    assert any("Qdrant" in context for context in contexts)


def test_rag_short_circuits_without_context(client, fake_generator) -> None:
    # Empty store: the search returns nothing, so the endpoint must skip generation.
    response = client.post(
        "/v1/rag", json={"query": "how to bake sourdough bread", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert "No relevant information" in body["answer"]
    assert body["sources"] == []


def test_rag_validates_query(client) -> None:
    response = client.post("/v1/rag", json={"query": ""})
    assert response.status_code == 422


def test_rag_excludes_low_confidence_hits(client, fake_generator, store, monkeypatch) -> None:
    from rag_platform.config.settings import get_settings
    from rag_platform.vectorstore.base import SearchHit

    settings = get_settings()
    monkeypatch.setattr(settings, "score_threshold", 0.9)
    monkeypatch.setattr(
        store,
        "search",
        lambda *args, **kwargs: [
            SearchHit(id="low-score", score=0.5, payload={"chunk_text": "irrelevant chunk"})
        ],
    )
    response = client.post("/v1/rag", json={"query": "anything", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert "No relevant information" in body["answer"]
    assert body["sources"] == []
    assert fake_generator.calls == []
