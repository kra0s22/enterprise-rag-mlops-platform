"""Hermetic tests for multi-tenancy isolation (Qdrant in-memory, no network)."""

from __future__ import annotations

import numpy as np

from rag_platform.api.tenancy import scope_filters
from rag_platform.utils.ids import make_chunk_id
from rag_platform.vectorstore.qdrant_store import QdrantStore


def _vector(dimension: int = 8) -> np.ndarray:
    vector = np.full(dimension, 1.0, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _store() -> QdrantStore:
    store = QdrantStore(path=":memory:", collection="test_tenancy", vector_size=8)
    store.create_collection()
    return store


def _seed(store: QdrantStore) -> None:
    tenant_a = [
        {"chunk_text": "alpha doc", "source": "a1.md", "tenant_id": "a"},
        {"chunk_text": "alpha doc two", "source": "a2.md", "tenant_id": "a"},
    ]
    tenant_b = [
        {"chunk_text": "beta doc", "source": "b1.md", "tenant_id": "b"},
    ]
    for index, payload in enumerate(tenant_a):
        store.upsert([make_chunk_id("a", index)], [_vector()], [payload])
    for index, payload in enumerate(tenant_b):
        store.upsert([make_chunk_id("b", index)], [_vector()], [payload])


def test_scope_filters_uses_default_tenant() -> None:
    assert scope_filters({}, None, "default") == {"tenant_id": "default"}


def test_scope_filters_preserves_request_filters() -> None:
    assert scope_filters({"source": "x.md"}, "acme", "default") == {
        "tenant_id": "acme",
        "source": "x.md",
    }


def test_scope_filters_explicit_beats_default() -> None:
    assert scope_filters({}, "acme", "default")["tenant_id"] == "acme"


def test_search_returns_only_requested_tenant() -> None:
    store = _store()
    _seed(store)

    hits_a = store.search(_vector(), top_k=10, filters={"tenant_id": "a"})
    hits_b = store.search(_vector(), top_k=10, filters={"tenant_id": "b"})

    assert {hit.payload["source"] for hit in hits_a} == {"a1.md", "a2.md"}
    assert {hit.payload["source"] for hit in hits_b} == {"b1.md"}
    assert all(hit.payload["tenant_id"] == "a" for hit in hits_a)
    assert all(hit.payload["tenant_id"] == "b" for hit in hits_b)


def test_search_without_tenant_scoping_sees_both() -> None:
    store = _store()
    _seed(store)

    hits = store.search(_vector(), top_k=10)
    assert len(hits) == 3
    assert {hit.payload["tenant_id"] for hit in hits} == {"a", "b"}


def test_api_scoped_filters_are_applied_end_to_end() -> None:
    """The API layer must scope with the effective tenant (helper contract)."""
    effective = scope_filters({}, tenant_id=None, default_tenant="acme")
    store = _store()
    _seed(store)
    # Only tenant "a" data exists; scoping to "acme" must yield no hits.
    assert store.search(_vector(), top_k=10, filters=effective) == []
