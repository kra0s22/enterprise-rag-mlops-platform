"""Token-based text chunking utilities.

Chunking is the backbone of retrieval quality: chunk size and overlap control the
granularity of indexed units and directly affect recall and precision in the vector
search. This module stays dependency-light and side-effect free so it can run both
locally and on PySpark executors.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace-delimited tokens."""
    if not text or not text.strip():
        return 0
    return len(_TOKEN_RE.findall(text))


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """Split ``text`` into overlapping token chunks.

    A sliding window over whitespace tokens guarantees every token is covered exactly
    once and consecutive chunks share ``chunk_overlap`` tokens, which reduces context
    loss at chunk boundaries during retrieval.

    Args:
        text: Input document text.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of tokens shared between consecutive chunks.

    Returns:
        List of chunk strings; empty list when the input is blank.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks
