"""Milvus-backed VectorStore implementation.

Dense vectors live in a ``FLOAT_VECTOR`` field (HNSW, COSINE); hybrid retrieval
also stores the hashing-based sparse representation in a ``SPARSE_FLOAT_VECTOR``
field and fuses both rankings with Milvus' reciprocal-rank fusion (``RRFRanker``).
Sparse vectors are exchanged as :class:`~rag_platform.embeddings.sparse.SparseVector`
and converted to Milvus' ``{index: value}`` dict format internally.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import SearchHit, VectorStore

logger = get_logger(__name__)

_DENSE_FIELD = "vector"
_SPARSE_FIELD = "sparse_vector"


def sparse_to_milvus(indices: list[int], values: list[float]) -> dict[int, float]:
    """Convert parallel index/value arrays to Milvus' ``{index: value}`` dict.

    Only non-zero values are emitted; Milvus sparse vectors are stored with inner
    product similarity, and the encoder already L2-normalizes the values.
    """
    return {
        int(index): float(value)
        for index, value in zip(indices, values, strict=True)
        if float(value) != 0.0
    }


class MilvusStore(VectorStore):
    """VectorStore implementation on top of Milvus (dense + sparse hybrid)."""

    supports_hybrid = True

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
        self._loaded = False

    def _connect(self) -> None:
        if self._connected:
            return
        kwargs: dict[str, Any] = {"uri": self._uri}
        if self._token:
            kwargs["token"] = self._token
        self._connections.connect(alias="default", **kwargs)
        self._connected = True

    def _get_collection(self) -> Any:
        """Return the collection, loading it into memory on first use.

        Milvus only serves ANN search on collections that have been loaded into
        the query nodes; ``load()`` is idempotent, so this is done once per store.
        """
        self._connect()
        collection = self._Collection(self._collection)
        if not self._loaded:
            collection.load()
            self._loaded = True
        return collection

    def _build_expr(self, filters: dict[str, Any] | None) -> str | None:
        """Compile payload filters into a Milvus boolean expression.

        Metadata lives in a JSON field, so filters become ``metadata["key"] == "value"``
        clauses (used for exact-match isolation such as tenant_id).
        """
        if not filters:
            return None
        clauses = []
        for key, value in filters.items():
            if isinstance(value, str):
                clauses.append(f'metadata["{key}"] == "{value}"')
            else:
                clauses.append(f'metadata["{key}"] == {value}')
        return " and ".join(clauses)

    def _to_hits(self, results: Any) -> list[SearchHit]:
        """Convert a pymilvus search/hybrid result set into SearchHit objects."""
        hits: list[SearchHit] = []
        for hit in results[0]:
            payload = dict(hit.entity.get("metadata") or {})
            payload["chunk_text"] = hit.entity.get("chunk_text", "")
            hits.append(SearchHit(id=str(hit.id), score=float(hit.score), payload=payload))
        return hits

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
                    name=_DENSE_FIELD,
                    dtype=self._DataType.FLOAT_VECTOR,
                    dim=self._vector_size,
                ),
                self._FieldSchema(name=_SPARSE_FIELD, dtype=self._DataType.SPARSE_FLOAT_VECTOR),
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
            field_name=_DENSE_FIELD,
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        collection.create_index(
            field_name=_SPARSE_FIELD,
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
            },
        )
        collection.load()
        self._loaded = True
        logger.info(
            "Created Milvus collection '%s' (dense=%s + sparse)",
            self._collection,
            self._vector_size,
        )

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        sparse_vectors: list[Any] | None = None,
    ) -> None:
        self._connect()
        collection = self._Collection(self._collection)
        rows = []
        for index, (pid, vector, payload) in enumerate(
            zip(ids, vectors, payloads, strict=True)
        ):
            row: dict[str, Any] = {
                "id": pid,
                _DENSE_FIELD: vector.tolist(),
                "chunk_text": payload.get("chunk_text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "chunk_text"},
            }
            if sparse_vectors is not None:
                sparse = sparse_vectors[index]
                row[_SPARSE_FIELD] = sparse_to_milvus(sparse.indices, sparse.values)
            rows.append(row)
        collection.insert(rows)
        collection.flush()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        self._connect()
        collection = self._get_collection()
        results = collection.search(
            data=[query_vector.tolist()],
            anns_field=_DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=self._build_expr(filters),
            output_fields=["id", "chunk_text", "metadata"],
        )
        return self._to_hits(results)

    def search_sparse(
        self,
        sparse_vector: Any,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Retrieve with the sparse representation only (used for sweeps)."""
        self._connect()
        collection = self._get_collection()
        results = collection.search(
            data=[sparse_to_milvus(sparse_vector.indices, sparse_vector.values)],
            anns_field=_SPARSE_FIELD,
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            output_fields=["id", "chunk_text", "metadata"],
        )
        return self._to_hits(results)

    def search_hybrid(
        self,
        dense_vector: np.ndarray,
        sparse_vector: Any,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Fuse dense and sparse rankings with Milvus reciprocal-rank fusion.

        The payload filter is applied per request (``hybrid_search`` has no
        ``expr`` argument in the client), so tenant scoping stays intact.
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        self._connect()
        collection = self._get_collection()
        expr = self._build_expr(filters)
        dense_request = AnnSearchRequest(
            data=[dense_vector.tolist()],
            anns_field=_DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_to_milvus(sparse_vector.indices, sparse_vector.values)],
            anns_field=_SPARSE_FIELD,
            param={"metric_type": "IP"},
            limit=top_k,
            expr=expr,
        )
        results = collection.hybrid_search(
            reqs=[dense_request, sparse_request],
            rerank=RRFRanker(k=60),
            limit=top_k,
            output_fields=["id", "chunk_text", "metadata"],
        )
        return self._to_hits(results)

    def delete(self, ids: list[str]) -> None:
        self._connect()
        collection = self._Collection(self._collection)
        collection.delete(f"id in {ids}")

    def count(self) -> int:
        self._connect()
        collection = self._Collection(self._collection)
        collection.flush()
        return int(collection.num_entities)
