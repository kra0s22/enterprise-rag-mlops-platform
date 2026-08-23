"""Milvus-backed VectorStore implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import SearchHit, VectorStore

logger = get_logger(__name__)


class MilvusStore(VectorStore):
    """VectorStore implementation on top of Milvus (HNSW, COSINE metric)."""

    def __init__(
        self,
        uri: str,
        collection: str = "documents",
        vector_size: int = 384,
        token: str | None = None,
    ) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

        self._connections = connections
        self._Collection = Collection
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        self._FieldSchema = FieldSchema
        self._uri = uri
        self._collection = collection
        self._vector_size = vector_size
        self._token = token
        self._connected = False

    def _connect(self) -> None:
        if self._connected:
            return
        kwargs: dict[str, Any] = {"uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        self._connections.connect(alias="default", **kwargs)
        self._connected = True

    def create_collection(self) -> None:
        from pymilvus import utility

        self._connect()
        if utility.has_collection(self._collection):
            return
        schema = self._CollectionSchema(
            fields=[
                self._FieldSchema(
                    name="id", dtype=self._DataType.VARCHAR, is_primary=True, max_length=128
                ),
                self._FieldSchema(
                    name="vector",
                    dtype=self._DataType.FLOAT_VECTOR,
                    dim=self._vector_size,
                ),
                self._FieldSchema(
                    name="chunk_text",
                    dtype=self._DataType.VARCHAR,
                    max_length=65535,
                ),
                self._FieldSchema(name="metadata", dtype=self._DataType.JSON),
            ]
        )
        collection = self._Collection(name=self._collection, schema=schema)
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        logger.info("Created Milvus collection '%s' (dim=%s)", self._collection, self._vector_size)

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        sparse_vectors: list[Any] | None = None,
    ) -> None:
        if sparse_vectors is not None:
            raise NotImplementedError("MilvusStore does not support hybrid (sparse) vectors")
        self._connect()
        collection = self._Collection(self._collection)
        rows = [
            {
                "id": pid,
                "vector": vector.tolist(),
                "chunk_text": payload.get("chunk_text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "chunk_text"},
            }
            for pid, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        collection.insert(rows)
        collection.flush()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        self._connect()
        collection = self._Collection(self._collection)
        expr = None
        if filters:
            expr = " and ".join(f'metadata["{k}"] == "{v}"' for k, v in filters.items())
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
            output_fields=["id", "chunk_text", "metadata"],
        )
        hits: list[SearchHit] = []
        for hit in results[0]:
            payload = dict(hit.entity.get("metadata") or {})
            payload["chunk_text"] = hit.entity.get("chunk_text", "")
            hits.append(SearchHit(id=str(hit.id), score=float(hit.score), payload=payload))
        return hits

    def delete(self, ids: list[str]) -> None:
        self._connect()
        collection = self._Collection(self._collection)
        collection.delete(f"id in {ids}")

    def count(self) -> int:
        self._connect()
        collection = self._Collection(self._collection)
        collection.flush()
        return int(collection.num_entities)
