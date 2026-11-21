from __future__ import annotations

import numpy as np
from openai import OpenAI

from rag_docs.embeddings.base import Vectors, l2_normalise


class OpenAIEmbedder:
    def __init__(
        self, model: str = "text-embedding-3-small", client: OpenAI | None = None, batch: int = 128
    ) -> None:
        self.model = model
        self.name = f"openai/{model}"
        self._client = client or OpenAI()
        self._batch = batch

    def embed_documents(self, texts: list[str]) -> Vectors:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            response = self._client.embeddings.create(
                model=self.model, input=texts[start : start + self._batch]
            )
            rows.extend(item.embedding for item in response.data)
        return l2_normalise(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> Vectors:
        return self.embed_documents([text])[0]
