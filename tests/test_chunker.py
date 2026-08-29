"""Unit tests for the token chunker."""

from __future__ import annotations

import pytest

from rag_platform.ingestion.chunker import (
    chunk_document,
    chunk_text,
    count_tokens,
    semantic_chunk_text,
    split_sections,
)


def test_count_tokens_counts_whitespace_delimited_tokens() -> None:
    assert count_tokens("one two three") == 3
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0


def test_chunk_text_single_chunk_when_short() -> None:
    text = "alpha beta gamma"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert chunks == ["alpha beta gamma"]


def test_chunk_text_chunks_respect_max_size() -> None:
    text = " ".join(f"token{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=32, chunk_overlap=8)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= 32


def test_chunk_text_overlap_shared_between_consecutive_chunks() -> None:
    text = " ".join(f"token{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 2
    for first, second in zip(chunks, chunks[1:], strict=False):
        first_tokens = first.split()
        second_tokens = second.split()
        # The first tokens of the second chunk must match the tail of the first chunk.
        assert first_tokens[-5:] == second_tokens[:5]


def test_chunk_text_fully_covers_input_tokens() -> None:
    text = " ".join(f"token{i}" for i in range(40))
    chunks = chunk_text(text, chunk_size=15, chunk_overlap=3)
    original_tokens = set(text.split())
    covered_tokens = set(" ".join(chunks).split())
    # Every original token must appear in at least one chunk.
    assert original_tokens <= covered_tokens


def test_chunk_text_blank_input_returns_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text(" \n\t ") == []


def test_chunk_text_validates_arguments() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=-1)


def test_split_sections_detects_headings() -> None:
    text = "# Intro\n\nlead text\n\n## One\n\nbody one\n\n## Two\n\nbody two"
    sections = split_sections(text)
    assert [heading for heading, _ in sections] == ["# Intro", "## One", "## Two"]
    assert "lead text" in sections[0][1]
    assert "body one" in sections[1][1]
    assert "body two" in sections[2][1]


def test_semantic_chunk_text_groups_small_sections() -> None:
    text = "# A\n\n" + "word " * 60 + "\n## B\n\n" + "foo " * 60
    chunks = semantic_chunk_text(text, chunk_size=200, chunk_overlap=10)
    assert len(chunks) == 1
    assert "# A" in chunks[0]
    assert "## B" in chunks[0]


def test_semantic_chunk_text_splits_oversized_section() -> None:
    text = "# A\n\n" + "word " * 500
    chunks = semantic_chunk_text(text, chunk_size=200, chunk_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= 200


def test_semantic_chunk_text_blank_input() -> None:
    assert semantic_chunk_text("") == []


def test_chunk_document_dispatches_modes() -> None:
    text = "# A\n\n" + "word " * 100 + "\n## B\n\n" + "foo " * 100
    assert chunk_document(text, 200, 10, mode="window") == chunk_text(text, 200, 10)
    assert chunk_document(text, 200, 10, mode="semantic") == semantic_chunk_text(text, 200, 10)


def test_chunk_document_invalid_mode() -> None:
    with pytest.raises(ValueError):
        chunk_document("hello", mode="bogus")
