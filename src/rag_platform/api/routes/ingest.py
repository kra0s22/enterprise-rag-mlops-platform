"""Document ingestion endpoints."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from rag_platform.api.dependencies import get_embedder, get_sparse_encoder, get_vector_store
from rag_platform.api.schemas import IngestRequest, IngestResponse
from rag_platform.config.settings import get_settings
from rag_platform.embeddings.provider import EmbeddingProvider
from rag_platform.embeddings.sparse import HashingSparseEncoder
from rag_platform.ingestion.chunker import chunk_text
from rag_platform.utils.ids import make_chunk_id
from rag_platform.utils.logging import get_logger
from rag_platform.vectorstore.base import VectorStore

router = APIRouter(prefix="/v1/ingest", tags=["ingestion"])
logger = get_logger(__name__)

StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
SparseDep = Annotated[HashingSparseEncoder, Depends(get_sparse_encoder)]


@router.post("", response_model=IngestResponse, status_code=201, summary="Ingest a document")
def ingest_document(
    request: IngestRequest,
    store: StoreDep,
    embedder: EmbedderDep,
    sparse_encoder: SparseDep,
) -> IngestResponse:
    """Chunk, embed, and store a document. Returns the ids of the stored chunks."""
    settings = get_settings()
    chunks = chunk_text(request.text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return IngestResponse(document_id="", num_chunks=0, stored_ids=[])

    document_id = request.document_id or uuid.uuid4().hex
    ids = [make_chunk_id(document_id, i) for i in range(len(chunks))]
    vectors = embedder.embed_documents(chunks)
    payloads = [
        {
            "chunk_text": chunk,
            "document_id": document_id,
            "chunk_index": i,
            **request.metadata,
        }
        for i, chunk in enumerate(chunks)
    ]
    store.upsert(
        ids,
        vectors,
        payloads,
        sparse_vectors=sparse_encoder.encode_batch(chunks),
    )
    logger.info("Ingested document '%s' (%d chunks)", document_id, len(chunks))
    return IngestResponse(document_id=document_id, num_chunks=len(chunks), stored_ids=ids)
