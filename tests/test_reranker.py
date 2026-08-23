"""Tests for the cross-encoder reranking helpers."""

from __future__ import annotations

from rag_platform.reranking.reranker import Reranker, rerank_hits
from rag_platform.vectorstore.base import SearchHit


class _FakeReranker(Reranker):
    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)

    def score(self, query: str, documents: list[str]) -> list[float]:
        return self._scores


def test_rerank_hits_reorders_and_updates_scores() -> None:
    hits = [
        SearchHit(id="a", score=0.1, payload={"chunk_text": "alpha"}),
        SearchHit(id="b", score=0.9, payload={"chunk_text": "beta"}),
        SearchHit(id="c", score=0.5, payload={"chunk_text": "gamma"}),
    ]

    reranked = rerank_hits(hits, "q", _FakeReranker([0.5, 1.0, 0.3]), top_k=2)

    assert [hit.id for hit in reranked] == ["b", "a"]
    assert reranked[0].score == 1.0
    assert reranked[1].score == 0.5


def test_rerank_hits_empty() -> None:
    assert rerank_hits([], "q", _FakeReranker([]), top_k=5) == []
