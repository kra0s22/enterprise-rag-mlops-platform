"""PySpark-based distributed document chunking pipeline.

The core function is pure and picklable so Spark can ship it to executors; each input
document row is expanded into ``N`` chunk rows, enabling fully distributed chunking of
large corpora.
"""

from __future__ import annotations

from typing import Any

from rag_platform.ingestion.chunker import chunk_document

_CHUNK_SCHEMA = (
    "document_id STRING, chunk_index INT, chunk_text STRING, "
    "metadata MAP<STRING, STRING>"
)


def _chunk_row(
    document_id: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, Any] | None,
    mode: str = "window",
) -> list[tuple[str, int, str, dict[str, str]]]:
    coerced = {str(k): str(v) for k, v in (metadata or {}).items()}
    return [
        (document_id, idx, chunk, dict(coerced))
        for idx, chunk in enumerate(
            chunk_document(text, chunk_size, chunk_overlap, mode=mode)
        )
    ]


def chunk_documents_spark(
    spark: Any,
    df: Any,
    chunk_size: int,
    chunk_overlap: int,
    mode: str = "window",
) -> Any:
    """Expand a DataFrame of documents into chunk rows via a PySpark UDF.

    The input DataFrame must contain columns ``document_id`` (str), ``text`` (str) and
    ``metadata`` (map<str,str>). Returns a DataFrame with one row per chunk and
    columns: ``document_id``, ``chunk_index``, ``chunk_text``, ``metadata``.

    Args:
        spark: Active SparkSession (used for config introspection).
        df: Input documents DataFrame.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of tokens shared between consecutive chunks.
        mode: Chunking mode, ``window`` or ``semantic``.

    Returns:
        A PySpark DataFrame of chunk rows.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType,
        IntegerType,
        MapType,
        StringType,
        StructField,
        StructType,
    )

    chunk_row_schema = StructType(
        [
            StructField("document_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk_text", StringType(), False),
            StructField("metadata", MapType(StringType(), StringType()), False),
        ]
    )

    chunk_udf = F.udf(
        lambda document_id, text, metadata: _chunk_row(
            document_id, text, int(chunk_size), int(chunk_overlap), metadata, mode
        ),
        ArrayType(chunk_row_schema),
    )

    expanded = df.withColumn(
        "_chunks", chunk_udf(F.col("document_id"), F.col("text"), F.col("metadata"))
    ).select(F.explode("_chunks").alias("chunk"))

    return expanded.select(
        F.col("chunk.document_id").alias("document_id"),
        F.col("chunk.chunk_index").alias("chunk_index"),
        F.col("chunk.chunk_text").alias("chunk_text"),
        F.col("chunk.metadata").alias("metadata"),
    )


__all__ = ["chunk_documents_spark", "_CHUNK_SCHEMA"]
