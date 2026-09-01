"""Hermetic tests for streaming ingestion (no network, no real Spark unless Java)."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

import rag_platform.ingestion.streaming as streaming


class _FakeEmbedder:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeStore:
    def __init__(self) -> None:
        self.upserted: list[tuple[list, list, list, list | None]] = []

    def upsert(
        self,
        ids: list[str],
        vectors: list,
        payloads: list[dict],
        sparse_vectors: list | None = None,
    ) -> None:
        self.upserted.append((list(ids), list(vectors), list(payloads), sparse_vectors))


class _FakeBatchDF:
    def __init__(self, rows: list) -> None:
        self._rows = rows
        self.sparkSession = object()

    def collect(self) -> list:
        return self._rows


def test_document_from_row_builds_deterministic_row() -> None:
    row = SimpleNamespace(source="data/sample/x.md", text="hello world")
    document = streaming.document_from_row(row)
    assert document["document_id"] == "b7d019fe7126b8a7"  # sha1("data/sample/x.md")[:16]
    assert document["text"] == "hello world"
    assert document["metadata"] == {"source": "data/sample/x.md"}


def test_document_from_row_same_source_same_id() -> None:
    first = streaming.document_from_row(SimpleNamespace(source="a.md", text="1"))
    second = streaming.document_from_row(SimpleNamespace(source="a.md", text="2"))
    assert first["document_id"] == second["document_id"]


def test_process_batch_empty_batch_returns_zero(monkeypatch) -> None:
    store = _FakeStore()
    assert (
        streaming.process_batch(
            _FakeBatchDF([]), _FakeEmbedder(), store, None, 512, 64
        )
        == 0
    )
    assert store.upserted == []


def test_process_batch_ingests_chunks(monkeypatch) -> None:
    store = _FakeStore()
    chunk_rows = [
        {"document_id": "d1", "chunk_index": 0, "chunk_text": "c0", "metadata": {"source": "x.md"}},
        {"document_id": "d1", "chunk_index": 1, "chunk_text": "c1", "metadata": {"source": "x.md"}},
    ]

    def fake_chunk(documents, spark, chunk_size, chunk_overlap, mode="window"):
        assert chunk_size == 512 and chunk_overlap == 64 and mode == "window"
        assert documents[0]["metadata"] == {"source": "x.md"}
        return chunk_rows

    monkeypatch.setattr(streaming, "_chunk_documents", fake_chunk)
    batch = _FakeBatchDF([SimpleNamespace(source="x.md", text="some content")])

    count = streaming.process_batch(
        batch, _FakeEmbedder(), store, None, 512, 64, mode="window"
    )

    assert count == 2
    assert len(store.upserted) == 1
    ids, vectors, payloads, sparse = store.upserted[0]
    assert len(ids) == 2
    assert payloads[0]["source"] == "x.md"
    assert sparse is None


def test_make_batch_processor_returns_callable(monkeypatch) -> None:
    store = _FakeStore()
    monkeypatch.setattr(
        streaming,
        "_chunk_documents",
        lambda documents, spark, *a, **k: [
            {
                "document_id": "d1",
                "chunk_index": 0,
                "chunk_text": "c0",
                "metadata": {"source": "x.md"},
            }
        ],
    )
    processor = streaming.make_batch_processor(
        _FakeEmbedder(), store, None, 512, 64, mode="window"
    )
    processor(_FakeBatchDF([SimpleNamespace(source="x.md", text="t")]), epoch_id=7)
    assert store.upserted  # one chunk upserted


@pytest.mark.skipif(shutil.which("java") is None, reason="requires a JVM for Spark")
def test_build_file_stream_creates_streaming_frame(tmp_path) -> None:
    """Functional check: the binaryFile source yields a streaming frame with the
    ``source``/``text`` columns. Runs in CI (ubuntu has Java) and in the Docker image."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-stream-schema")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        stream = streaming.build_file_stream(spark, str(watch_dir))
        assert stream.isStreaming
        assert set(stream.columns) == {"source", "text"}
    finally:
        spark.stop()
