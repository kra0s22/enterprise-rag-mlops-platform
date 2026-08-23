"""Cross-encoder reranking to improve retrieval quality.

A cross-encoder scores each (query, candidate) pair jointly, which is more
accurate than the bi-encoder similarity used at retrieval time. The heavy model
is imported lazily so serving paths that never rerank stay lightweight.
"""

from __future__ import annotations

from rag_platform.vectorstore.base import SearchHit


class Reranker:
    """Contract for rerankers; score each query-document pair."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return a relevance score per document (higher is better)."""
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Rerank candidates with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, document) for document in documents]
        return [float(value) for value in self._model.predict(pairs)]


def rerank_hits(
    hits: list[SearchHit],
    query: str,
    reranker: Reranker,
    top_k: int,
) -> list[SearchHit]:
    """Re-score ``hits`` with ``reranker`` and return the ``top_k``.

    The returned hits carry the reranker score in ``SearchHit.score``.
    """
    if not hits:
        return []
    texts = [str(hit.payload.get("chunk_text", "")) for hit in hits]
    scores = reranker.score(query, texts)
    ordered = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    top = ordered[:top_k]
    for hit, score in top:
        hit.score = score
    return [hit for hit, _ in top]
