"""Document loading from local files (txt, markdown, pdf)."""

from __future__ import annotations

from pathlib import Path

from rag_platform.utils.logging import get_logger

logger = get_logger(__name__)

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


class DocumentLoadError(RuntimeError):
    """Raised when a document cannot be read."""


def load_text_file(path: Path) -> str:
    """Read a UTF-8 text file, replacing undecodable bytes."""
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: Path) -> str:
    """Extract text from a PDF document."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError("pypdf is required to load PDF documents") from exc
    if not path.is_file():
        raise DocumentLoadError(f"File not found: {path}")
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_document(path: Path) -> str:
    """Load a document into plain text based on its file extension."""
    ext = path.suffix.lower()
    if ext in _TEXT_EXTENSIONS:
        return load_text_file(path)
    if ext == ".pdf":
        return load_pdf_file(path)
    raise DocumentLoadError(f"Unsupported file extension: {ext}")
