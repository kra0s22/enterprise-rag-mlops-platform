"""Retrieval endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from rag_platform.api.dependencies import get_embedder, get_sparse_encoder, get_vector_store
from rag_platform.api.schemas import SearchHitResponse, SearchRequest, SearchResponse
from rag_platform.embeddings.provider import EmbeddingProvider
from rag_platform.embeddings.sparse import HashingSparseEncoder
from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import VectorStore

router = APIRouter(prefix="/v1/search", tags=["retrieval"])
logger = get_logger(__name__)

StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
SparseDep = Annotated[HashingSparseEncoder, Depends(get_sparse_encoder)]


@router.post("", response_model=SearchResponse, summary="Search the vector store")
def search(
    request: SearchRequest,
    store: StoreDep,
    embedder: EmbedderDep,
    sparse_encoder: SparseDep,
) -> SearchResponse:
    """Embed the query and return the top-k nearest chunks."""
    if request.hybrid:
        if not store.supports_hybrid:
            raise HTTPException(
                status_code=400,
                detail="Hybrid search is not supported by the configured vector store",
            )
        hits = store.search_hybrid(
            embedder.embed_query(request.query),
            sparse_encoder.encode(request.query),
            top_k=request.top_k,
            filters=request.filters,
        )
    else:
        query_vector = embedder.embed_query(request.query)
        hits = store.search(query_vector, top_k=request.top_k, filters=request.filters)
    return SearchResponse(
        query=request.query,
        hits=[
            SearchHitResponse(
                id=hit.id,
                score=hit.score,
                chunk_text=str(hit.payload.get("chunk_text", "")),
                metadata=hit.payload,
            )
            for hit in hits
        ],
    )
