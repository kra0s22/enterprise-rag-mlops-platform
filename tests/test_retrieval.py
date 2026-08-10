"""Integration-style tests for vector retrieval (Qdrant in-memory + fake embedder)."""

from __future__ import annotations

import uuid

import pytest

from rag_platform.vectorstore.base import SearchHit


@pytest.fixture()
def populated_store(store, embedder):
    """A store with a few deterministic chunks ingested."""
    texts = [
        "the cat sat on the mat",
        "quantum physics explains the very small",
        "the dog chased the ball across the park",
        "machine learning models need large datasets",
    ]
    ids = [str(uuid.uuid4()) for _ in texts]
    vectors = embedder.embed_documents(texts)
    payloads = [
        {"chunk_text": t, "document_id": f"doc{i}", "chunk_index": 0, "category": "general"}
        for i, t in enumerate(texts)
    ]
    store.upsert(ids, vectors, payloads)
    return store, texts, ids


def test_upsert_and_count(populated_store) -> None:
    store, texts, ids = populated_store
    assert store.count() == len(texts)


def test_exact_match_is_top_hit(populated_store, embedder) -> None:
    store, texts, ids = populated_store
    query = "the cat sat on the mat"
    hits = store.search(embedder.embed_query(query), top_k=1)
    assert len(hits) == 1
    assert hits[0].payload["chunk_text"] == query


def test_top_k_respected(populated_store, embedder) -> None:
    store, texts, ids = populated_store
    hits = store.search(embedder.embed_query("animal behaviour"), top_k=3)
    assert len(hits) <= 3
    assert all(isinstance(h, SearchHit) for h in hits)
    # Scores must be non-increasing.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_with_payload_filter(populated_store, embedder) -> None:
    store, texts, ids = populated_store
    hits = store.search(embedder.embed_query("anything"), top_k=10, filters={"category": "general"})
    assert all(h.payload.get("category") == "general" for h in hits)


def test_delete_removes_vectors(populated_store, embedder) -> None:
    store, texts, ids = populated_store
    store.delete([ids[0]])
    assert store.count() == len(texts) - 1
    hits = store.search(embedder.embed_query("the cat sat on the mat"), top_k=1)
    assert hits[0].payload["chunk_text"] != "the cat sat on the mat"
