"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

RRF only uses ranks, so it needs no score calibration between a cosine
similarity and a BM25 score, and it is robust to either retriever failing.
"""

from __future__ import annotations

from typing import Literal

from rag_docs.embeddings.base import Embedder
from rag_docs.models import ScoredChunk
from rag_docs.retrieval.bm25 import BM25Index
from rag_docs.stores.base import VectorStore

Mode = Literal["dense", "bm25", "hybrid"]


def reciprocal_rank_fusion(rankings: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(hit.chunk.id, hit)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [ScoredChunk(chunk=by_id[cid].chunk, score=scores[cid]) for cid in ordered]


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, mode: Mode = "hybrid") -> None:
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self._bm25: BM25Index | None = None

    def refresh(self) -> None:
        """Rebuild the lexical index after the corpus changes."""
        self._bm25 = BM25Index(self.store.all_chunks())

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        if self._bm25 is None:
            self.refresh()
        assert self._bm25 is not None
        dense = (
            self.store.search(self.embedder.embed_query(query), k) if self.mode != "bm25" else []
        )
        lexical = self._bm25.search(query, k) if self.mode != "dense" else []
        if self.mode == "dense":
            return dense
        if self.mode == "bm25":
            return lexical
        return reciprocal_rank_fusion([dense, lexical])[:k]
