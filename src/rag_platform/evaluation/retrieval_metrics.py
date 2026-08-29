"""Chunk-level retrieval metrics (MRR@k, hit-rate@k, nDCG@k).

These are pure, deterministic metrics over ranked lists, independent of any LLM
or generator: they measure how well the vector store surfaces the relevant chunks
for a query. They are the retrieval-stage counterpart to the end-to-end Ragas
metrics and let an A/B run attribute gains to the retrieval stack.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

ChunkKey = tuple[str, int]


def _is_relevant(relevant: Iterable[ChunkKey], item: ChunkKey) -> bool:
    return item in relevant


def mrr_at_k(
    relevant: Iterable[ChunkKey],
    ranked: Sequence[ChunkKey],
    k: int,
) -> float:
    """Return the reciprocal rank of the first relevant chunk within ``top_k``.

    0.0 when no relevant chunk appears in the first ``k`` positions.
    """
    relevant_set = set(relevant)
    for index, item in enumerate(ranked[:k]):
        if _is_relevant(relevant_set, item):
            return 1.0 / (index + 1)
    return 0.0


def hit_rate_at_k(
    relevant: Iterable[ChunkKey],
    ranked: Sequence[ChunkKey],
    k: int,
) -> float:
    """Return 1.0 if any relevant chunk is in the ``top_k``, else 0.0."""
    relevant_set = set(relevant)
    return 1.0 if any(_is_relevant(relevant_set, item) for item in ranked[:k]) else 0.0


def ndcg_at_k(
    relevant: Iterable[ChunkKey],
    ranked: Sequence[ChunkKey],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain with binary relevance.

    Uses the log2 discount typical of information retrieval; 0.0 when there are
    no relevant chunks or ``k`` is zero.
    """
    relevant_set = set(relevant)
    if not relevant_set or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, item in enumerate(ranked[:k])
        if _is_relevant(relevant_set, item)
    )
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevant_set), k)))
    return dcg / ideal if ideal > 0.0 else 0.0


def chunk_indices_for_keywords(
    text: str,
    keywords: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
    mode: str = "window",
) -> list[int]:
    """Return the chunk indices of ``text`` that contain any ``keyword``.

    Chunks are produced with the same ``chunk_size``/``chunk_overlap``/``mode``
    used at ingestion, so relevance stays aligned with the indexed collection
    across chunking configurations (needed for chunk-size and chunk-mode
    ablations). A keyword is a case-insensitive substring of the chunk.
    """
    from rag_platform.ingestion.chunker import chunk_document

    if not keywords:
        return []
    chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, mode=mode)
    lowered = [keyword.lower() for keyword in keywords]
    return [
        index
        for index, chunk in enumerate(chunks)
        if any(keyword in chunk.lower() for keyword in lowered)
    ]
