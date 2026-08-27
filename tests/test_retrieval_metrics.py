"""Hermetic tests for chunk-level retrieval metrics."""

from __future__ import annotations

import math

from rag_platform.evaluation.retrieval_metrics import (
    ChunkKey,
    chunk_indices_for_keywords,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
)

RANKED: list[ChunkKey] = [("a.md", 0), ("b.md", 0), ("c.md", 0), ("d.md", 0)]
RELEVANT = {("b.md", 0)}


def test_mrr_first_position() -> None:
    assert mrr_at_k({("a.md", 0)}, RANKED, k=3) == 1.0


def test_mrr_second_position() -> None:
    assert math.isclose(mrr_at_k(RELEVANT, RANKED, k=3), 0.5)


def test_mrr_beyond_cutoff() -> None:
    assert mrr_at_k(RELEVANT, RANKED, k=1) == 0.0


def test_mrr_no_relevant() -> None:
    assert mrr_at_k({("z.md", 9)}, RANKED, k=3) == 0.0


def test_hit_rate_within_cutoff() -> None:
    assert hit_rate_at_k(RELEVANT, RANKED, k=3) == 1.0


def test_hit_rate_miss() -> None:
    assert hit_rate_at_k(RELEVANT, RANKED, k=1) == 0.0


def test_hit_rate_irrelevant_query() -> None:
    assert hit_rate_at_k({("z.md", 9)}, RANKED, k=3) == 0.0


def test_ndcg_perfect_ranking() -> None:
    assert math.isclose(ndcg_at_k({("a.md", 0)}, RANKED, k=3), 1.0)


def test_ndcg_imperfect_ranking() -> None:
    # Relevant at position 2 -> DCG 1/log2(3); ideal 1.0.
    assert math.isclose(ndcg_at_k(RELEVANT, RANKED, k=3), 1.0 / math.log2(3))


def test_ndcg_empty_relevant() -> None:
    assert ndcg_at_k(set(), RANKED, k=3) == 0.0


def test_ndcg_zero_cutoff() -> None:
    assert ndcg_at_k(RELEVANT, RANKED, k=0) == 0.0


TEXT = " ".join(f"word{i}" for i in range(100)) + " signal watermark tail"


def test_keywords_find_matching_chunk() -> None:
    indices = chunk_indices_for_keywords(TEXT, ["watermark"], chunk_size=30, chunk_overlap=5)
    assert indices == [3]


def test_keywords_absent_returns_empty() -> None:
    assert chunk_indices_for_keywords(TEXT, ["nonexistent"], 30, 5) == []


def test_keywords_case_insensitive() -> None:
    assert chunk_indices_for_keywords("Hello World", ["hello"], 10, 0) == [0]


def test_keywords_empty_list() -> None:
    assert chunk_indices_for_keywords(TEXT, [], 30, 5) == []
