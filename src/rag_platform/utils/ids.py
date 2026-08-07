"""Identifier helpers for vector store points."""

from __future__ import annotations

import uuid


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Build a stable, valid UUID point id for a chunk.

    Qdrant requires string point ids to be valid UUIDs, so we derive a
    deterministic UUID from the logical ``(document_id, chunk_index)`` pair.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))
