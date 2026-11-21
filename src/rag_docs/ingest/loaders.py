"""Turn source files into normalised, page-level text."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from pypdf import PdfReader

from rag_docs.models import Page

_WHITESPACE = re.compile("[ \\t\\u00a0]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def document_id_for(content: bytes) -> str:
    """Content-addressed id: re-uploading the same file never duplicates chunks."""
    return hashlib.sha256(content).hexdigest()[:16]


def normalise(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_pdf(content: bytes, source: str) -> list[Page]:
    reader = PdfReader(io.BytesIO(content))
    doc_id = document_id_for(content)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = normalise(page.extract_text() or "")
        if text:
            pages.append(Page(document_id=doc_id, source=source, page_number=number, text=text))
    return pages


def load_text(content: bytes, source: str) -> list[Page]:
    """Plain text and Markdown are treated as a single page."""
    text = normalise(content.decode("utf-8", errors="replace"))
    if not text:
        return []
    return [Page(document_id=document_id_for(content), source=source, page_number=1, text=text)]


def load_path(path: Path) -> list[Page]:
    content = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        return load_pdf(content, path.name)
    return load_text(content, path.name)
