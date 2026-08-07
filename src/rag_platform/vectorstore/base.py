"""Abstract vector store interface for the RAG platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SearchHit:
    """A single retrieval result."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Contract implemented by every supported vector database backend.

    Vectors are passed as ``np.ndarray`` (shape ``(dimension,)`` for a single vector
    or ``(N, dimension)`` for batches) so the API layer stays backend-agnostic.
    """

    @abstractmethod
    def create_collection(self) -> None:
        """Create the collection/space if it does not exist."""

    @abstractmethod
    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        """Insert or update vectors together with their payloads."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the ``top_k`` nearest neighbours of ``query_vector``.

        ``filters`` maps payload field names to exact-match values.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by id."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""

    def healthcheck(self) -> bool:
        """Lightweight liveness probe; subclasses may override."""
        try:
            self.count()
            return True
        except Exception:  # noqa: BLE001
            return False
