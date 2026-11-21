"""Chroma-backed store, used by the docker compose setup."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from rag_docs.embeddings.base import Vectors
from rag_docs.models import Chunk, ScoredChunk


class ChromaVectorStore:
    def __init__(self, url: str, collection: str = "rag_docs") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install the chroma extra: uv sync --extra chroma") from exc
        parsed = urlparse(url)
        client = chromadb.HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000)
        self._col: Any = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: Vectors) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {"document_id": c.document_id, "source": c.source, "page_number": c.page_number}
                for c in chunks
            ],
        )

    def search(self, query_vector: Vectors, k: int) -> list[ScoredChunk]:
        res = self._col.query(query_embeddings=[query_vector.tolist()], n_results=k)
        out = []
        for cid, text, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            chunk = Chunk(id=cid, text=text, **meta)
            out.append(ScoredChunk(chunk=chunk, score=1.0 - float(dist)))
        return out

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        return [
            Chunk(id=cid, text=text, **meta)
            for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def delete_document(self, document_id: str) -> int:
        ids = self._col.get(where={"document_id": document_id})["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)
