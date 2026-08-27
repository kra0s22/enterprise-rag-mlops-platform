"""Hermetic tests for client-side reciprocal rank fusion."""

from __future__ import annotations

from rag_platform.vectorstore.base import SearchHit
from rag_platform.vectorstore.fusion import keys_from_hits, rrf_fuse


def test_rrf_common_top_item_wins() -> None:
    dense = ["a", "b", "c"]
    sparse = ["a", "c", "b"]
    # "a" ranks 1 in both -> highest fused score.
    assert rrf_fuse([dense, sparse], k=60)[0] == "a"


def test_rrf_order_respects_fused_scores() -> None:
    # "x" top in one list only; "a" top in both -> "a" before "x".
    fused = rrf_fuse([["a", "b", "x"], ["a", "x", "b"]], k=60)
    assert fused.index("a") < fused.index("x")


def test_rrf_larger_k_flattens() -> None:
    high = rrf_fuse([["a", "b"], ["a", "b"]], k=1)
    low = rrf_fuse([["a", "b"], ["a", "b"]], k=100)
    assert high == low  # ties resolved consistently; ordering by fused score


def test_rrf_weights_dominate() -> None:
    # Second ranking has "z" first and dominates via a large weight.
    fused = rrf_fuse([["a", "z"], ["z", "a"]], k=10, weights=[0.1, 10.0])
    assert fused[0] == "z"


def test_rrf_empty_rankings() -> None:
    assert rrf_fuse([], k=60) == []


def test_rrf_invalid_k() -> None:
    try:
        rrf_fuse([["a"]], k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for k=0")


def test_rrf_weights_length_mismatch() -> None:
    try:
        rrf_fuse([["a"], ["b"]], k=60, weights=[1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for weight mismatch")


def test_keys_from_hits_builds_stable_keys() -> None:
    hits = [
        SearchHit(id="1", score=0.5, payload={"source": "a.md", "chunk_index": 0}),
        SearchHit(id="2", score=0.4, payload={"source": "b.md", "chunk_index": 2}),
    ]
    assert keys_from_hits(hits) == ["a.md#0", "b.md#2"]
