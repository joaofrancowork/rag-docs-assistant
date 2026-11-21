from __future__ import annotations

from typing import TYPE_CHECKING

from rag_docs.stores.base import VectorStore
from rag_docs.stores.memory import InMemoryVectorStore

if TYPE_CHECKING:
    from rag_docs.config import Settings

__all__ = ["InMemoryVectorStore", "VectorStore", "build_store"]


def build_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "chroma":  # pragma: no cover - exercised via docker compose
        from rag_docs.stores.chroma import ChromaVectorStore

        return ChromaVectorStore(settings.chroma_url)
    return InMemoryVectorStore(persist_dir=settings.index_dir)
