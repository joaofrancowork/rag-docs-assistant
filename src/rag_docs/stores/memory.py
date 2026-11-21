"""Exact cosine search with NumPy, persisted to disk.

For a few thousand pages a brute-force matrix product is faster and simpler
than an ANN index; swap in Chroma (or pgvector) when the corpus grows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_docs.embeddings.base import Vectors
from rag_docs.models import Chunk, ScoredChunk


class InMemoryVectorStore:
    def __init__(self, persist_dir: Path | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: Vectors | None = None
        self._dir = persist_dir
        if persist_dir and (persist_dir / "chunks.json").exists():
            self._load()

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: Vectors) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        existing = {c.id for c in self._chunks}
        keep = [i for i, c in enumerate(chunks) if c.id not in existing]
        if not keep:
            return
        self._chunks.extend(chunks[i] for i in keep)
        new = vectors[keep]
        self._vectors = new if self._vectors is None else np.vstack([self._vectors, new])
        self._save()

    def search(self, query_vector: Vectors, k: int) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            return []
        scores = self._vectors @ query_vector
        top = np.argsort(-scores)[:k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def delete_document(self, document_id: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep] if keep and self._vectors is not None else None
            self._save()
        return removed

    def _save(self) -> None:
        if not self._dir:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self._chunks]), encoding="utf-8"
        )
        if self._vectors is not None:
            np.save(self._dir / "vectors.npy", self._vectors)
        else:
            (self._dir / "vectors.npy").unlink(missing_ok=True)

    def _load(self) -> None:
        assert self._dir is not None
        raw = json.loads((self._dir / "chunks.json").read_text(encoding="utf-8"))
        self._chunks = [Chunk.model_validate(c) for c in raw]
        vectors_path = self._dir / "vectors.npy"
        self._vectors = np.load(vectors_path) if vectors_path.exists() else None
