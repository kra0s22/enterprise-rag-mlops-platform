"""Integration tests against a live stack (API + Qdrant + Ollama).

These tests are safe for the default CI run: they skip unless
``RAG_RUN_INTEGRATION=1`` is set, so ``pytest`` stays deterministic and
offline. When enabled, they exercise the real retrieval and generation path
through the HTTP API end to end.

Run locally with the stack up:

    $env:RAG_RUN_INTEGRATION = "1"
    python -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os

import httpx
import pytest

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
TOP_K = 3
QUERY = "What is Qdrant used for in this platform?"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RAG_RUN_INTEGRATION") != "1",
        reason="Integration tests require RAG_RUN_INTEGRATION=1 and a live stack",
    ),
]


def _search(**overrides: object) -> dict:
    payload = {"query": QUERY, "top_k": TOP_K, **overrides}
    response = httpx.post(f"{API_URL}/v1/search", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def test_health() -> None:
    response = httpx.get(f"{API_URL}/health", timeout=10)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_dense_returns_grounded_hits() -> None:
    body = _search()
    assert len(body["hits"]) >= 1
    hit = body["hits"][0]
    assert hit["id"]
    assert hit["score"] is not None
    assert hit["chunk_text"]
    assert hit["metadata"]["source"]


def test_hybrid_fuses_sparse_retrieval() -> None:
    dense_scores = [hit["score"] for hit in _search()["hits"]]
    hybrid_scores = [hit["score"] for hit in _search(hybrid=True)["hits"]]
    # RRF fusion must change the ranking signal vs pure dense cosine similarity.
    assert hybrid_scores != dense_scores


def test_rerank_rescores_with_cross_encoder() -> None:
    hybrid_scores = [hit["score"] for hit in _search(hybrid=True)["hits"]]
    rerank_scores = [hit["score"] for hit in _search(hybrid=True, rerank=True)["hits"]]
    # The cross-encoder must replace the fusion scores with its own logits.
    assert rerank_scores != hybrid_scores


def test_rag_returns_grounded_answer() -> None:
    payload = {"query": QUERY, "top_k": TOP_K}
    response = httpx.post(f"{API_URL}/v1/rag", json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    assert body["answer"]
    assert len(body["sources"]) >= 1
