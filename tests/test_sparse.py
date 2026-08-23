"""Tests for the sparse encoder and hybrid (dense + sparse) retrieval."""

from __future__ import annotations

import math
import uuid

from rag_platform.embeddings.sparse import HashingSparseEncoder
from rag_platform.vectorstore.base import VectorStore


def test_sparse_encoder_is_deterministic_and_l2_normalized() -> None:
    encoder = HashingSparseEncoder(n_features=64)
    first = encoder.encode("quantum physics")
    second = encoder.encode("quantum physics")
    assert first.indices == second.indices
    assert first.values == second.values
    norm = math.sqrt(sum(value * value for value in first.values))
    assert abs(norm - 1.0) < 1e-6
    assert len(first.indices) == len(first.values)
    assert max(first.indices, default=-1) < 64


def test_sparse_encoder_shares_index_space() -> None:
    encoder = HashingSparseEncoder(n_features=64)
    query = encoder.encode("quantum physics")
    document = encoder.encode("quantum physics explains the very small")
    assert set(query.indices) & set(document.indices)  # overlapping tokens align


def test_hybrid_search_fuses_dense_and_sparse(store, embedder) -> None:
    texts = [
        "the cat sat on the mat",
        "quantum physics explains the very small",
        "machine learning models need large datasets",
    ]
    ids = [str(uuid.uuid4()) for _ in texts]
    dense = embedder.embed_documents(texts)
    encoder = HashingSparseEncoder(n_features=64)
    store.upsert(
        ids,
        dense,
        [{"chunk_text": t} for t in texts],
        sparse_vectors=encoder.encode_batch(texts),
    )

    hits = store.search_hybrid(
        embedder.embed_query("cat mat"), encoder.encode("cat mat"), top_k=3
    )

    assert len(hits) == 3
    assert hits[0].payload["chunk_text"] == "the cat sat on the mat"
    assert hits[0].score >= hits[-1].score


def test_qdrant_store_advertises_hybrid_support(store) -> None:
    assert store.supports_hybrid is True
    assert isinstance(store, VectorStore)


def test_base_search_hybrid_raises_for_unsupported_backend() -> None:
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

    store = _DenseOnly()
    try:
        store.search_hybrid([0.1] * 8, None)
    except NotImplementedError as exc:
        assert "does not support hybrid search" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
