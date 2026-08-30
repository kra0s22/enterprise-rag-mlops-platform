"""Hermetic tests for HyDE query expansion."""

from __future__ import annotations

import numpy as np

from rag_platform.generation.hyde import hyde_query_vector


class _FakeClient:
    def generate_hypothesis(self, query: str) -> str:
        return f"hypothetical passage answering {query}"


class _FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.array([float(len(text)), 0.0], dtype=np.float32)


def test_hyde_query_vector_embeds_hypothesis() -> None:
    query = "question?"
    vector = hyde_query_vector(query, _FakeClient(), _FakeEmbedder())
    # The embedded text is the hypothetical passage, which is longer than the
    # raw query, so its first component must be larger.
    assert vector[0] > float(len(query))
