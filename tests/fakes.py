"""Deterministic fake embedding provider for tests (no model download, no network)."""

from __future__ import annotations

import hashlib

import numpy as np

from rag_platform.embeddings.provider import EmbeddingProvider


class DummyEmbeddingProvider(EmbeddingProvider):
    """Hash-based provider producing stable pseudo-random vectors per text."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        vector = np.resize(raw, self._dimension)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return self._dimension
