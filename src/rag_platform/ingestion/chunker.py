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


_HEADING_RE = re.compile(r"^#{1,6}\s+.+$")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into ``(heading, body)`` sections by markdown headings.

    Text before the first heading becomes a section with an empty heading, so
    plain-text documents are returned as a single section.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        if _HEADING_RE.match(line):
            if heading or body:
                sections.append((heading, "\n".join(body)))
            heading = line.strip()
            body = []
        else:
            body.append(line)
    if heading or body:
        sections.append((heading, "\n".join(body)))
    return sections


def semantic_chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Structure-aware chunking that keeps markdown sections together.

    Consecutive sections are grouped while they fit under ``chunk_size``; a single
    section larger than ``chunk_size`` is split with the token window so no chunk
    exceeds the limit. Keeps the heading as context inside each resulting chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for heading, body in split_sections(text):
        section_text = f"{heading}\n{body}".strip() if heading else body.strip()
        if not section_text:
            continue
        section_tokens = count_tokens(section_text)
        if buffer_tokens + section_tokens <= chunk_size:
            buffer.append(section_text)
            buffer_tokens += section_tokens
        else:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_tokens = [], 0
            if section_tokens > chunk_size:
                chunks.extend(chunk_text(section_text, chunk_size, chunk_overlap))
            else:
                buffer.append(section_text)
                buffer_tokens = section_tokens
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


CHUNK_MODES = ("window", "semantic")


def chunk_document(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    mode: str = "window",
) -> list[str]:
    """Chunk a document with the selected ``mode`` (``window`` or ``semantic``)."""
    if mode == "semantic":
        return semantic_chunk_text(text, chunk_size, chunk_overlap)
    if mode == "window":
        return chunk_text(text, chunk_size, chunk_overlap)
    raise ValueError(f"Unknown chunk mode: {mode!r} (expected one of {CHUNK_MODES})")
