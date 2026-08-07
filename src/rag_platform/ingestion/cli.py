"""Command-line entry point for the batch ingestion pipeline.

Usage:
    rag-ingest ./data/sample ./extra/report.pdf
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from rag_platform.config.settings import get_settings
from rag_platform.embeddings.provider import build_embedding_provider
from rag_platform.ingestion.chunker import chunk_text
from rag_platform.ingestion.loader import DocumentLoadError, load_document
from rag_platform.utils.ids import make_chunk_id
from rag_platform.utils.logging import configure_logging, get_logger
from rag_platform.vectorstore.factory import build_vector_store

logger = get_logger(__name__)

_SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}


def _collect_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED
        )
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-ingest documents into the vector store.")
    parser.add_argument("paths", nargs="+", type=Path, help="Document files or directories")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings, embedder.dimension)
    store.create_collection()

    total = 0
    for entry in args.paths:
        for file in _collect_files(entry):
            try:
                text = load_document(file)
            except DocumentLoadError as exc:
                logger.error("Skipping %s: %s", file, exc)
                continue

            document_id = hashlib.sha1(str(file).encode("utf-8")).hexdigest()[:16]
            chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                continue

            ids = [make_chunk_id(document_id, i) for i in range(len(chunks))]
            vectors = embedder.embed_documents(chunks)
            payloads = [
                {
                    "chunk_text": chunk,
                    "document_id": document_id,
                    "chunk_index": i,
                    "source": str(file),
                }
                for i, chunk in enumerate(chunks)
            ]
            store.upsert(ids, vectors, payloads)
            total += len(chunks)
            logger.info("Ingested %s (%d chunks)", file, len(chunks))

    logger.info("Ingestion finished: %d chunks upserted", total)


if __name__ == "__main__":
    main()
