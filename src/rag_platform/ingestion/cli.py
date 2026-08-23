"""Command-line entry point for the batch ingestion pipeline.

Usage:
    rag-ingest ./data/sample ./extra/report.pdf
    rag-ingest --distributed ./data/corpus
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_platform.config.settings import get_settings
from rag_platform.embeddings.provider import build_embedding_provider
from rag_platform.ingestion.chunker import chunk_text
from rag_platform.ingestion.loader import DocumentLoadError, load_document
from rag_platform.utils.ids import make_chunk_id
from rag_platform.utils.logging import configure_logging, get_logger
from rag_platform.vectorstore.base import VectorStore
from rag_platform.vectorstore.factory import build_vector_store

logger = get_logger(__name__)

_SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}


def _collect_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED
        )
    return [path]


def load_documents(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load every supported document into a row for the ingestion pipeline.

    Each row carries a deterministic ``document_id``, the raw ``text`` and a
    ``metadata`` mapping. Documents that fail to load are logged and skipped.
    """
    rows: list[dict[str, Any]] = []
    for entry in paths:
        for file in _collect_files(entry):
            try:
                text = load_document(file)
            except DocumentLoadError as exc:
                logger.error("Skipping %s: %s", file, exc)
                continue
            document_id = hashlib.sha1(str(file).encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "document_id": document_id,
                    "text": text,
                    "metadata": {"source": str(file)},
                }
            )
    return rows


def ingest_chunks(embedder: Any, store: VectorStore, chunks: Sequence[dict[str, Any]]) -> int:
    """Embed and upsert chunk rows; returns the number of chunks stored.

    Each chunk row carries ``document_id``, ``chunk_index``, ``chunk_text`` and a
    ``metadata`` mapping that is merged into the vector payload.
    """
    if not chunks:
        return 0
    ids = [make_chunk_id(chunk["document_id"], chunk["chunk_index"]) for chunk in chunks]
    vectors = embedder.embed_documents([chunk["chunk_text"] for chunk in chunks])
    payloads = [
        {
            "chunk_text": chunk["chunk_text"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            **chunk["metadata"],
        }
        for chunk in chunks
    ]
    store.upsert(ids, vectors, payloads)
    return len(chunks)


def _distributed_chunks(
    documents: Sequence[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    spark: Any | None = None,
) -> list[dict[str, Any]]:
    """Chunk documents in parallel with PySpark and collect the chunk rows.

    ``spark`` is injected in tests; when omitted a local SparkSession is started
    (requires a JDK reachable via ``JAVA_HOME``).
    """
    from pyspark.sql import SparkSession

    from rag_platform.ingestion.spark_pipeline import chunk_documents_spark

    session = spark or SparkSession.builder.master("local[*]").appName("rag-ingest").getOrCreate()
    docs_df = session.createDataFrame(list(documents))
    chunk_df = chunk_documents_spark(session, docs_df, chunk_size, chunk_overlap)
    return [
        {
            "document_id": row.document_id,
            "chunk_index": int(row.chunk_index),
            "chunk_text": row.chunk_text,
            "metadata": dict(row.metadata),
        }
        for row in chunk_df.collect()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-ingest documents into the vector store.")
    parser.add_argument("paths", nargs="+", type=Path, help="Document files or directories")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Chunk documents with a local PySpark session instead of single-process chunking",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings, embedder.dimension)
    store.create_collection()

    documents = load_documents(args.paths)
    if not documents:
        logger.warning("No supported documents found; nothing to ingest")
        return

    if args.distributed:
        chunks = _distributed_chunks(documents, settings.chunk_size, settings.chunk_overlap)
        total = ingest_chunks(embedder, store, chunks)
        logger.info("Distributed ingestion finished: %d chunks upserted", total)
        return

    total = 0
    for doc in documents:
        chunks = [
            {
                "document_id": doc["document_id"],
                "chunk_index": index,
                "chunk_text": chunk,
                "metadata": dict(doc["metadata"]),
            }
            for index, chunk in enumerate(
                chunk_text(doc["text"], settings.chunk_size, settings.chunk_overlap)
            )
        ]
        if chunks:
            total += ingest_chunks(embedder, store, chunks)
            logger.info("Ingested %s (%d chunks)", doc["metadata"]["source"], len(chunks))

    logger.info("Ingestion finished: %d chunks upserted", total)


if __name__ == "__main__":
    main()
