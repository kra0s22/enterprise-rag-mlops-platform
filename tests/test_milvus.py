"""Hermetic tests for the Milvus hybrid backend (no Milvus server needed).

The pure conversion and filter-expression helpers are covered here; full search
round-trips require a live Milvus and are validated in the integration layer.
"""

from __future__ import annotations

import pytest

from rag_platform.vectorstore.milvus_store import MilvusStore, sparse_to_milvus


def test_sparse_to_milvus_converts_parallel_arrays() -> None:
    assert sparse_to_milvus([1, 5, 9], [0.5, 0.0, 0.25]) == {1: 0.5, 9: 0.25}


def test_sparse_to_milvus_preserves_all_nonzero() -> None:
    result = sparse_to_milvus([0, 3, 7], [1.0, -0.5, 0.2])
    assert result == {0: 1.0, 3: -0.5, 7: 0.2}


def test_sparse_to_milvus_empty() -> None:
    assert sparse_to_milvus([], []) == {}


def test_sparse_to_milvus_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        sparse_to_milvus([1, 2], [0.5])


def test_milvus_store_supports_hybrid() -> None:
    store = MilvusStore(uri="http://localhost:19530", collection="x", vector_size=4)
    assert store.supports_hybrid is True


def test_build_expr_string_filter() -> None:
    store = MilvusStore(uri="http://localhost:19530", collection="x", vector_size=4)
    assert store._build_expr({"tenant_id": "a"}) == 'metadata["tenant_id"] == "a"'


def test_build_expr_combines_filters() -> None:
    store = MilvusStore(uri="http://localhost:19530", collection="x", vector_size=4)
    expr = store._build_expr({"tenant_id": "a", "category": "general"})
    assert 'metadata["tenant_id"] == "a"' in expr
    assert 'metadata["category"] == "general"' in expr


def test_build_expr_numeric_filter() -> None:
    store = MilvusStore(uri="http://localhost:19530", collection="x", vector_size=4)
    assert store._build_expr({"chunk_index": 2}) == 'metadata["chunk_index"] == 2'


def test_build_expr_none_without_filters() -> None:
    store = MilvusStore(uri="http://localhost:19530", collection="x", vector_size=4)
    assert store._build_expr(None) is None
    assert store._build_expr({}) is None
