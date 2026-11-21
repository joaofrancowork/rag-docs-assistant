"""Embedding providers behind one small protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_docs.embeddings.base import Embedder
from rag_docs.embeddings.hashing import HashingEmbedder

if TYPE_CHECKING:
    from rag_docs.config import Settings

__all__ = ["Embedder", "HashingEmbedder", "build_embedder"]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    if settings.embedding_provider == "openai":
        from rag_docs.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model=settings.embedding_model or "text-embedding-3-small")
    from rag_docs.embeddings.fastembed import FastEmbedEmbedder  # pragma: no cover

    return FastEmbedEmbedder(  # pragma: no cover
        model=settings.embedding_model or "BAAI/bge-small-en-v1.5"
    )
