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

    supports_hybrid: bool = False

    @abstractmethod
    def create_collection(self) -> None:
        """Create the collection/space if it does not exist."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        sparse_vectors: Any | None = None,
    ) -> None:
        """Insert or update vectors together with their payloads.

        ``sparse_vectors`` optionally carries the sparse representation of the
        same points for hybrid-capable backends; backends without hybrid
        support reject a non-empty value.
        """

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

    def search_hybrid(
        self,
        dense_vector: np.ndarray,
        sparse_vector: Any,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Fuse dense and sparse retrieval into a single ranked result.

        Only backends with ``supports_hybrid`` override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support hybrid search")

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
