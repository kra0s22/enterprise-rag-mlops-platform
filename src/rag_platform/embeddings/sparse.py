"""Sparse (hashing-based) text encoder for hybrid retrieval.

A neural SPLADE encoder would require a model download; a feature-hashing
encoder (``HashingVectorizer``) is dependency-free, deterministic and produces
vectors compatible with Qdrant's native sparse vectors. The class exposes the
same ``encode``/``encode_batch`` surface so it can be swapped for SPLADE later
without touching call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import HashingVectorizer


@dataclass(slots=True)
class SparseVector:
    """A sparse vector as a pair of parallel index/value arrays."""

    indices: list[int]
    values: list[float]


class HashingSparseEncoder:
    """Encode text into L2-normalized hashed sparse vectors.

    Queries and documents share the same vectorizer, so the index space is
    aligned across the corpus (required for nearest-neighbour search).
    """

    def __init__(self, n_features: int = 4096) -> None:
        self.n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            norm="l2",
            alternate_sign=False,
        )

    def encode(self, text: str) -> SparseVector:
        """Return a deterministic sparse vector for ``text``."""
        coo = self._vectorizer.transform([text]).tocoo()
        order = coo.col.argsort()
        return SparseVector(
            indices=coo.col[order].tolist(),
            values=coo.data[order].tolist(),
        )

    def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        """Return the sparse vectors for a batch of texts."""
        return [self.encode(text) for text in texts]
