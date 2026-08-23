"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Payload for the document ingestion endpoint."""

    text: str = Field(..., min_length=1, description="Document text to ingest")
    document_id: str | None = Field(default=None, description="Optional stable document id")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary document metadata"
    )


class IngestResponse(BaseModel):
    """Result of a document ingestion."""

    document_id: str
    num_chunks: int
    stored_ids: list[str]


class SearchRequest(BaseModel):
    """Payload for the retrieval endpoint."""

    query: str = Field(..., min_length=1, description="Natural-language query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of hits to return")
    filters: dict[str, Any] = Field(default_factory=dict, description="Exact-match payload filters")
    hybrid: bool = Field(default=False, description="Use hybrid dense+sparse retrieval")


class SearchHitResponse(BaseModel):
    """A single retrieval hit."""

    id: str
    score: float
    chunk_text: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Result of a retrieval request."""

    query: str
    hits: list[SearchHitResponse]


class RagRequest(BaseModel):
    """Payload for the grounded generation endpoint."""

    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Exact-match payload filters"
    )
    hybrid: bool = Field(default=False, description="Use hybrid dense+sparse retrieval")


class RagSource(BaseModel):
    """A context chunk used to ground the generated answer."""

    id: str
    score: float
    chunk_text: str
    metadata: dict[str, Any]


class RagResponse(BaseModel):
    """Result of a grounded generation request."""

    query: str
    answer: str
    sources: list[RagSource]
