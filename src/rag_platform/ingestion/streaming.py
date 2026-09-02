"""Streaming ingestion with Spark Structured Streaming.

Watches a directory and continuously indexes new files. The ``binaryFile`` source
yields every new file as a micro-batch with columns ``path``, ``modificationTime``,
``length`` and ``content``; the content is decoded to text, chunks are produced with
the same distributed UDF as the batch path, and ``foreachBatch`` embeds and upserts
them to the vector store from the driver. ``foreachBatch`` is the standard escape
hatch for sinks Spark does not know about, such as a vector database.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from rag_platform.ingestion.cli import ingest_chunks
from rag_platform.ingestion.spark_pipeline import chunk_documents_spark
from rag_platform.utils.logging import get_logger

logger = get_logger(__name__)

_STREAM_SCHEMA = "source STRING, text STRING"


def build_file_stream(spark: Any, watch_dir: str, max_files_per_trigger: int = 10) -> Any:
    """Return a streaming DataFrame of new files under ``watch_dir``.

    Each record exposes ``source`` (the file path) and ``text`` (the decoded file
    content); the ``binaryFile`` format handles arbitrary text/markdown files that
    a schema-based source (csv/json) cannot. The source schema is declared
    explicitly because a streaming reader cannot infer it from existing files.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        BinaryType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    file_schema = StructType(
        [
            StructField("path", StringType(), False),
            StructField("modificationTime", TimestampType(), False),
            StructField("length", LongType(), False),
            StructField("content", BinaryType(), False),
        ]
    )
    return (
        spark.readStream.format("binaryFile")
        .schema(file_schema)
        .option("maxFilesPerTrigger", max_files_per_trigger)
        .load(watch_dir)
        .select(
            F.col("path").alias("source"),
            F.col("content").cast("string").alias("text"),
        )
    )


def document_from_row(row: Any) -> dict[str, Any]:
    """Convert a streaming row (``source``, ``text``) into an ingestion document.

    The ``document_id`` is derived from the file path exactly like the batch path
    (sha1 prefix), so re-ingesting the same file overwrites the same chunk ids.
    """
    source = str(row.source)
    return {
        "document_id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:16],
        "text": str(row.text),
        "metadata": {"source": source},
    }


def _chunk_documents(
    documents: list[dict[str, Any]],
    spark: Any,
    chunk_size: int,
    chunk_overlap: int,
    mode: str,
) -> list[dict[str, Any]]:
    """Chunk document dicts with the distributed Spark UDF and collect the rows.

    The documents carry ``document_id``/``text``/``metadata`` (the columns the UDF
    expects), built from the streaming rows the same way the batch path builds them.
    """
    docs_df = spark.createDataFrame(list(documents))
    chunk_df = chunk_documents_spark(spark, docs_df, chunk_size, chunk_overlap, mode=mode)
    return [
        {
            "document_id": row.document_id,
            "chunk_index": int(row.chunk_index),
            "chunk_text": row.chunk_text,
            "metadata": dict(row.metadata),
        }
        for row in chunk_df.collect()
    ]


def process_batch(
    batch_df: Any,
    embedder: Any,
    store: Any,
    sparse_encoder: Any,
    chunk_size: int,
    chunk_overlap: int,
    mode: str = "window",
    tenant_id: str | None = None,
) -> int:
    """Ingest one micro-batch of documents; returns the number of chunks stored.

    ``batch_df`` must expose ``sparkSession`` and the columns ``source``/``text``.
    Used as the ``foreachBatch`` sink so tests can drive it with a fake batch.
    """
    rows = batch_df.collect()
    if not rows:
        logger.debug("Skipping empty streaming batch")
        return 0
    documents = [document_from_row(row) for row in rows]
    chunks = _chunk_documents(
        documents, batch_df.sparkSession, chunk_size, chunk_overlap, mode
    )
    count = ingest_chunks(
        embedder, store, chunks, sparse_encoder=sparse_encoder, tenant_id=tenant_id
    )
    logger.info(
        "Ingested %d chunks from %d new file(s) in the streaming batch",
        count,
        len(documents),
    )
    return count


def make_batch_processor(
    embedder: Any,
    store: Any,
    sparse_encoder: Any,
    chunk_size: int,
    chunk_overlap: int,
    mode: str = "window",
    tenant_id: str | None = None,
) -> Callable[[Any, int], None]:
    """Return a ``foreachBatch``-compatible callable for a streaming query."""

    def process(batch_df: Any, epoch_id: int) -> None:
        process_batch(
            batch_df,
            embedder,
            store,
            sparse_encoder,
            chunk_size,
            chunk_overlap,
            mode=mode,
            tenant_id=tenant_id,
        )
        logger.debug("Streaming epoch %s processed", epoch_id)

    return process


def run_streaming_ingestion(
    spark: Any,
    watch_dir: str,
    checkpoint_dir: str,
    processor: Callable[[Any, int], None],
    trigger_ms: int = 10_000,
    max_files_per_trigger: int = 10,
) -> None:
    """Start a structured streaming query and block until it is terminated.

    ``checkpoint_dir`` persists offsets and state so restarts resume without
    re-processing already-ingested files (fault tolerance).
    """
    stream = build_file_stream(spark, watch_dir, max_files_per_trigger)
    query = (
        stream.writeStream.foreachBatch(processor)
        .option("checkpointLocation", checkpoint_dir)
        .trigger(processingTime=f"{trigger_ms} milliseconds")
        .start()
    )
    logger.info("Streaming ingestion started on %s (checkpoint: %s)", watch_dir, checkpoint_dir)
    query.awaitTermination()
