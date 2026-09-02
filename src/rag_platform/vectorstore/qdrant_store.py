"""Qdrant-backed VectorStore implementation.

The dense vector is stored under the named vector ``dense`` so a future sparse vector
(``sparse``) can be added for hybrid search without breaking existing data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import SearchHit, VectorStore

logger = get_logger(__name__)

_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"


class QdrantStore(VectorStore):
    """VectorStore implementation on top of Qdrant.

    Pass ``path=":memory:"`` (or a file path) for a local, server-less instance —
    used by the hermetic test suite.
    """

    supports_hybrid = True

    def __init__(
        self,
        url: str | None = None,
        collection: str = "documents",
        vector_size: int = 384,
        api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        if path is None:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(path=path)
        self._collection = collection
        self._vector_size = vector_size

    def create_collection(self) -> None:
        try:
            self._client.get_collection(self._collection)
            return
        except Exception:  # noqa: BLE001 - collection missing
            pass
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE_NAME: self._models.VectorParams(
                    size=self._vector_size,
                    distance=self._models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                _SPARSE_NAME: self._models.SparseVectorParams(),
            },
        )
        # Every chunk carries tenant_id (multi-tenancy); an index on the field
        # keeps tenant-scoped filtered searches fast on a shared collection.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="tenant_id",
            field_schema=self._models.PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "Created Qdrant collection '%s' (dense=%s, sparse=%s)",
            self._collection,
            self._vector_size,
            _SPARSE_NAME,
        )

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        sparse_vectors: list[Any] | None = None,
    ) -> None:
        if sparse_vectors is None:
            points = [
                self._models.PointStruct(
                    id=pid,
                    vector={_DENSE_NAME: vector.tolist()},
                    payload=payload,
                )
                for pid, vector, payload in zip(ids, vectors, payloads, strict=True)
            ]
        else:
            points = [
                self._models.PointStruct(
                    id=pid,
                    vector={
                        _DENSE_NAME: vector.tolist(),
                        _SPARSE_NAME: self._models.SparseVector(
                            indices=sparse.indices, values=sparse.values
                        ),
                    },
                    payload=payload,
                )
                for pid, vector, sparse, payload in zip(
                    ids, vectors, sparse_vectors, payloads, strict=True
                )
            ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        query_filter = None
        if filters:
            query_filter = self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key=key,
                        match=self._models.MatchValue(value=value),
                    )
                    for key, value in filters.items()
                ]
            )
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            using=_DENSE_NAME,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(id=str(hit.id), score=float(hit.score), payload=dict(hit.payload or {}))
            for hit in response.points
        ]

    def search_hybrid(
        self,
        dense_vector: np.ndarray,
        sparse_vector: Any,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Retrieve with dense and sparse queries and fuse via reciprocal rank."""
        query_filter = None
        if filters:
            query_filter = self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key=key,
                        match=self._models.MatchValue(value=value),
                    )
                    for key, value in filters.items()
                ]
            )
        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                self._models.Prefetch(
                    query=dense_vector.tolist(), using=_DENSE_NAME, limit=top_k
                ),
                self._models.Prefetch(
                    query=self._models.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                    using=_SPARSE_NAME,
                    limit=top_k,
                ),
            ],
            query=self._models.FusionQuery(fusion=self._models.Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(id=str(hit.id), score=float(hit.score), payload=dict(hit.payload or {}))
            for hit in response.points
        ]

    def search_sparse(
        self,
        sparse_vector: Any,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Retrieve with the sparse vector only (used for fusion sweeps)."""
        response = self._client.query_points(
            collection_name=self._collection,
            query=self._models.SparseVector(
                indices=sparse_vector.indices, values=sparse_vector.values
            ),
            using=_SPARSE_NAME,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(id=str(hit.id), score=float(hit.score), payload=dict(hit.payload or {}))
            for hit in response.points
        ]

    def delete(self, ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._models.PointIdsList(points=ids),
        )

    def count(self) -> int:
        return int(self._client.count(collection_name=self._collection).count)

    def iter_points(self) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
        """Scroll the whole collection and return ``(id, dense, payload)``.

        Drift monitoring snapshots the embedding distribution of the corpus, so it
        needs to read every stored point rather than only query results.
        """
        records: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        offset: Any = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_vectors=True,
            )
            for point in points:
                dense = (
                    point.vector.get(_DENSE_NAME)
                    if isinstance(point.vector, dict)
                    else point.vector
                )
                records.append(
                    (
                        str(point.id),
                        np.asarray(dense, dtype=np.float32),
                        dict(point.payload or {}),
                    )
                )
            if next_offset is None:
                return records
            offset = next_offset
