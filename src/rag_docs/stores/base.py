from __future__ import annotations

from typing import Protocol

from rag_docs.embeddings.base import Vectors
from rag_docs.models import Chunk, ScoredChunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: Vectors) -> None: ...

    def search(self, query_vector: Vectors, k: int) -> list[ScoredChunk]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def delete_document(self, document_id: str) -> int: ...
