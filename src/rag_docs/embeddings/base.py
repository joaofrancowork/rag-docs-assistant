from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    """Maps texts to L2-normalised vectors, so a dot product is cosine similarity."""

    name: str

    def embed_documents(self, texts: list[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


def l2_normalise(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
