"""Grounded generation endpoint (retrieve + generate with Ollama)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from rag_platform.api.dependencies import get_embedder, get_vector_store
from rag_platform.api.schemas import RagRequest, RagResponse, RagSource
from rag_platform.config.settings import get_settings
from rag_platform.embeddings.provider import EmbeddingProvider
from rag_platform.generation.client import build_generation_client
from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import VectorStore

router = APIRouter(prefix="/v1/rag", tags=["generation"])
logger = get_logger(__name__)

StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]


@router.post("", response_model=RagResponse, summary="Generate a grounded answer")
def generate_answer(
    request: RagRequest,
    store: StoreDep,
    embedder: EmbedderDep,
) -> RagResponse:
    """Retrieve the top-k chunks for ``query`` and generate a grounded answer with Ollama."""
    settings = get_settings()
    client = build_generation_client(settings)

    query_vector = embedder.embed_query(request.query)
    hits = store.search(query_vector, top_k=request.top_k, filters=request.filters)
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
