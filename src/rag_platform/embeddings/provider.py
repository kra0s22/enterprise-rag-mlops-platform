"""Embedding provider abstraction over local Sentence-Transformers models.

The heavy ``torch``/``sentence_transformers`` imports are deferred so the FastAPI
serving import path stays fast and dependency-light.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EmbeddingProvider(ABC):
    """Contract for generating dense embedding vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return a dense matrix of shape ``(len(texts), dimension)``."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return a dense vector of shape ``(dimension,)``."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider backed by a SentenceTransformer model."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, deferred

            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        self._load()
        return self._dimension or 0


def build_embedding_provider(settings: Any) -> EmbeddingProvider:
    """Instantiate the embedding provider from application settings."""
    return SentenceTransformerProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
