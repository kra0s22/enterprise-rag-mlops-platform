"""Hermetic tests for chunk-level retrieval metrics."""

from __future__ import annotations

import math

from rag_platform.evaluation.retrieval_metrics import (
    ChunkKey,
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
