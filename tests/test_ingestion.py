"""Hermetic tests for the ingestion CLI helpers (local and distributed paths)."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

import rag_platform.ingestion.cli as cli


class _FakeEmbedder:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
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


def test_load_documents_reads_supported_files(tmp_path) -> None:
    doc = tmp_path / "guide.md"
    doc.write_text("# Heading\n\nBody text.", encoding="utf-8")

    rows = cli.load_documents([tmp_path])

    assert len(rows) == 1
    assert rows[0]["text"] == "# Heading\n\nBody text."
    assert rows[0]["metadata"]["source"] == str(doc)
    assert len(rows[0]["document_id"]) == 16  # sha1 prefix


def test_load_documents_filters_unsupported_extensions(tmp_path) -> None:
    (tmp_path / "notes.log").write_text("ignored", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("kept", encoding="utf-8")

    rows = cli.load_documents([tmp_path])

    assert [r["metadata"]["source"] for r in rows] == [str(tmp_path / "readme.txt")]


def test_ingest_chunks_embeds_and_upserts() -> None:
    embedder = _FakeEmbedder()
    store = _FakeStore()
    chunks = [
        {"document_id": "d1", "chunk_index": 0, "chunk_text": "c0", "metadata": {"source": "s"}},
        {"document_id": "d1", "chunk_index": 1, "chunk_text": "c1", "metadata": {"source": "s"}},
    ]

    count = cli.ingest_chunks(embedder, store, chunks)

    assert count == 2
    assert len(store.upserted) == 1
    ids, vectors, payloads, sparse_vectors = store.upserted[0]
    assert len(ids) == 2 and len(vectors) == 2
    assert sparse_vectors is None
    assert payloads[0]["source"] == "s"
    assert payloads[0]["chunk_text"] == "c0"


def test_ingest_chunks_empty_returns_zero() -> None:
    store = _FakeStore()
    assert cli.ingest_chunks(_FakeEmbedder(), store, []) == 0
    assert store.upserted == []


def test_ingest_chunks_with_sparse_encoder_passes_sparse_vectors() -> None:
    from rag_platform.embeddings.sparse import HashingSparseEncoder

    store = _FakeStore()
    chunks = [
        {
            "document_id": "d1",
            "chunk_index": 0,
            "chunk_text": "quantum physics",
            "metadata": {"source": "s"},
        }
    ]
    cli.ingest_chunks(
        _FakeEmbedder(),
        store,
        chunks,
        sparse_encoder=HashingSparseEncoder(n_features=32),
    )
    _, _, _, sparse_vectors = store.upserted[0]
    assert sparse_vectors is not None
    assert len(sparse_vectors) == 1
    assert sparse_vectors[0].indices  # non-empty sparse vector


def test_distributed_chunks_collects_spark_rows(monkeypatch) -> None:
    import rag_platform.ingestion.spark_pipeline as spark_pipeline

    class _FakeChunkDF:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def collect(self) -> list[SimpleNamespace]:
            return self._rows

    class _FakeSpark:
        def __init__(self, df: _FakeChunkDF) -> None:
            self._df = df

        def createDataFrame(self, documents: list[dict]) -> _FakeChunkDF:
            return self._df

    rows = [
        SimpleNamespace(document_id="d1", chunk_index=0, chunk_text="c0", metadata={"source": "s"}),
        SimpleNamespace(document_id="d1", chunk_index=1, chunk_text="c1", metadata={"source": "s"}),
    ]

    def fake_chunk(spark, df, chunk_size: int, chunk_overlap: int) -> _FakeChunkDF:
        assert chunk_size == 512 and chunk_overlap == 64
        return _FakeChunkDF(rows)

    monkeypatch.setattr(spark_pipeline, "chunk_documents_spark", fake_chunk)
    spark = _FakeSpark(_FakeChunkDF(rows))

    out = cli._distributed_chunks(
        [{"document_id": "d1", "text": "t", "metadata": {"source": "s"}}],
        512,
        64,
        spark=spark,
    )

    assert out == [
        {"document_id": "d1", "chunk_index": 0, "chunk_text": "c0", "metadata": {"source": "s"}},
        {"document_id": "d1", "chunk_index": 1, "chunk_text": "c1", "metadata": {"source": "s"}},
    ]


@pytest.mark.skipif(shutil.which("java") is None, reason="Java/JDK not installed")
def test_distributed_chunks_runs_with_real_spark() -> None:
    """Functional: chunk a document with a real local SparkSession (needs Java)."""
    rows = cli._distributed_chunks(
        [{"document_id": "d1", "text": "word " * 700, "metadata": {"source": "s"}}],
        100,
        10,
    )
    assert len(rows) > 1
    for row in rows:
        assert row["document_id"] == "d1"
        assert row["metadata"] == {"source": "s"}
        assert row["chunk_text"]
