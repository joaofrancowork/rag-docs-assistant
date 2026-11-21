"""Okapi BM25 over chunks.

Dense retrieval misses exact identifiers (policy numbers, product codes, names);
BM25 catches them. Small enough to keep in memory and rebuild on ingest.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rag_docs.models import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it of on or "
    "the to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks = chunks
        self._tfs = [Counter(tokenize(c.text)) for c in chunks]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lengths) / len(chunks)) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tfs):
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / (self._avg_len or 1))
            score = sum(
                self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [ScoredChunk(chunk=self._chunks[i], score=s) for s, i in scored[:k]]
