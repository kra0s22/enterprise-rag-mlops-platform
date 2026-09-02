"""Grounded generation endpoint (retrieve + generate with Ollama)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from rag_platform.api.dependencies import (
    get_embedder,
    get_reranker,
    get_sparse_encoder,
    get_vector_store,
)
from rag_platform.api.schemas import RagRequest, RagResponse, RagSource
from rag_platform.api.tenancy import scope_filters
from rag_platform.config.settings import get_settings
from rag_platform.embeddings.provider import EmbeddingProvider
from rag_platform.embeddings.sparse import HashingSparseEncoder
from rag_platform.generation.client import build_generation_client
from rag_platform.reranking.reranker import Reranker, rerank_hits
from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import VectorStore

router = APIRouter(prefix="/v1/rag", tags=["generation"])
logger = get_logger(__name__)

StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
SparseDep = Annotated[HashingSparseEncoder, Depends(get_sparse_encoder)]
RerankDep = Annotated[Reranker, Depends(get_reranker)]


@router.post("", response_model=RagResponse, summary="Generate a grounded answer")
def generate_answer(
    request: RagRequest,
    store: StoreDep,
    embedder: EmbedderDep,
    sparse_encoder: SparseDep,
    reranker: RerankDep,
) -> RagResponse:
    """Retrieve the top-k chunks for ``query`` and generate a grounded answer with Ollama."""
    settings = get_settings()
    client = build_generation_client(settings)

    retrieve_k = (
        (request.rerank_top_k or settings.rerank_top_k)
        if request.rerank
        else request.top_k
    )
    query_vector = embedder.embed_query(request.query)
    if request.hyde:
        from rag_platform.generation.hyde import hyde_query_vector

        query_vector = hyde_query_vector(request.query, client, embedder)
    if request.hybrid:
        if not store.supports_hybrid:
            raise HTTPException(
                status_code=400,
                detail="Hybrid search is not supported by the configured vector store",
            )
        hits = store.search_hybrid(
            query_vector,
            sparse_encoder.encode(request.query),
            top_k=retrieve_k,
            filters=scope_filters(request.filters, request.tenant_id, settings.tenant),
        )
    else:
        hits = store.search(
            query_vector,
            top_k=retrieve_k,
            filters=scope_filters(request.filters, request.tenant_id, settings.tenant),
        )
    hits = [hit for hit in hits if hit.score >= settings.score_threshold]
    if request.rerank:
        hits = rerank_hits(hits, request.query, reranker, request.top_k)
    if not hits:
        logger.info("No context retrieved for query=%r; skipping generation", request.query)
        return RagResponse(
            query=request.query,
            answer="No relevant information was found in the knowledge base for this question.",
            sources=[],
        )
    contexts = [str(hit.payload.get("chunk_text", "")) for hit in hits]

    answer = client.generate(request.query, contexts)

    sources = [
        RagSource(id=hit.id, score=hit.score, chunk_text=context, metadata=hit.payload)
        for hit, context in zip(hits, contexts, strict=True)
    ]
    return RagResponse(query=request.query, answer=answer, sources=sources)
