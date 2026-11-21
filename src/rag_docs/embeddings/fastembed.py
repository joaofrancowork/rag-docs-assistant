"""Local ONNX embeddings (no API key), via the optional ``fastembed`` extra."""

from __future__ import annotations

import numpy as np

from rag_docs.embeddings.base import Vectors, l2_normalise


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("Install the local extra: uv sync --extra local") from exc
        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed/{model}"

    def embed_documents(self, texts: list[str]) -> Vectors:
        return l2_normalise(np.asarray(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> Vectors:
        vectors = l2_normalise(np.asarray(list(self._model.query_embed(text)), dtype=np.float32))
        vector: Vectors = vectors[0]
        return vector
