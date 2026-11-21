"""Deterministic, dependency-free embedder.

Feature-hashes word unigrams and character trigrams into a fixed-size vector.
It is not semantic, but it is fast, offline and reproducible, which makes it
ideal for tests and CI, and a useful lexical baseline in evaluations.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from rag_docs.embeddings.base import Vectors, l2_normalise

_TOKEN = re.compile(r"[a-z0-9]+")


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        trigrams = [f"#{w[i : i + 3]}" for w in words for i in range(max(1, len(w) - 2))]
        return words + trigrams

    def embed_documents(self, texts: list[str]) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                index, sign = _bucket(feature, self.dim)
                out[row, index] += sign
        return l2_normalise(out)

    def embed_query(self, text: str) -> Vectors:
        vector: Vectors = self.embed_documents([text])[0]
        return vector
