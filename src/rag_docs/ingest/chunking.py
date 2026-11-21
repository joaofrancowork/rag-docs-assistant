"""Structure-aware chunking.

Splits on the largest natural boundary that keeps a chunk under ``chunk_size``
(paragraph, then sentence, then word), and carries a character overlap between
neighbouring chunks so that facts spanning a boundary stay retrievable.
Chunks never cross page boundaries, which keeps citations exact.
"""

from __future__ import annotations

import re

from rag_docs.models import Chunk, Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    for pattern in ("\n\n", _SENTENCE_END, " "):
        parts = text.split(pattern) if isinstance(pattern, str) else pattern.split(text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            pieces: list[str] = []
            for part in parts:
                pieces.extend(_split(part, size))
            return pieces
    # A single token longer than the chunk size: hard cut.
    return [text[i : i + size] for i in range(0, len(text), size)]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # Start the overlap at a word boundary.
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{tail} {piece}".strip() if tail else piece
            if len(current) > size:
                current = piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for page in pages:
        texts = _merge(_split(page.text, chunk_size), chunk_size, overlap)
        for i, text in enumerate(texts):
            chunks.append(
                Chunk(
                    id=f"{page.document_id}:{page.page_number}:{i}",
                    document_id=page.document_id,
                    source=page.source,
                    page_number=page.page_number,
                    text=text,
                )
            )
    return chunks
