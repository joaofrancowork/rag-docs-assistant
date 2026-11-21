from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_docs.config import Settings
from rag_docs.embeddings import HashingEmbedder, build_embedder
from rag_docs.embeddings.openai import OpenAIEmbedder
from rag_docs.models import Chunk
from rag_docs.stores import InMemoryVectorStore, build_store


def _chunk(cid: str, text: str, doc: str = "d1") -> Chunk:
    return Chunk(id=cid, document_id=doc, source=f"{doc}.pdf", page_number=1, text=text)


def test_hashing_embedder_is_normalised_and_deterministic() -> None:
    emb = HashingEmbedder(dim=128)
    a = emb.embed_documents(["refund policy", "refund policy"])
    assert a.shape == (2, 128)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a[0], a[1])


def test_hashing_embedder_ranks_lexical_overlap_higher() -> None:
    emb = HashingEmbedder()
    docs = emb.embed_documents(["how to request a refund", "support opening hours"])
    query = emb.embed_query("refund request")
    scores = docs @ query
    assert scores[0] > scores[1]


def test_empty_text_gives_zero_vector_without_nan() -> None:
    vec = HashingEmbedder(dim=16).embed_query("")
    assert not np.isnan(vec).any()


def test_openai_embedder_batches_and_normalises() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = lambda model, input: MagicMock(
        data=[MagicMock(embedding=[3.0, 4.0]) for _ in input]
    )
    emb = OpenAIEmbedder(client=client, batch=2)
    out = emb.embed_documents(["a", "b", "c"])
    assert client.embeddings.create.call_count == 2
    assert np.allclose(out, [[0.6, 0.8]] * 3)
    assert np.allclose(emb.embed_query("x"), [0.6, 0.8])


def test_build_embedder_from_settings() -> None:
    assert isinstance(build_embedder(Settings(embedding_provider="hashing")), HashingEmbedder)


def test_store_search_dedupe_and_delete() -> None:
    emb = HashingEmbedder()
    store = InMemoryVectorStore()
    chunks = [_chunk("1", "refund within 30 days"), _chunk("2", "office in Lisbon", doc="d2")]
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    store.add(chunks[:1], emb.embed_documents([chunks[0].text]))  # duplicate ignored
    assert len(store) == 2

    hits = store.search(emb.embed_query("refund"), k=1)
    assert hits[0].chunk.id == "1"

    assert store.delete_document("d1") == 1
    assert [c.id for c in store.all_chunks()] == ["2"]
    assert store.delete_document("missing") == 0
    assert store.delete_document("d2") == 1
    assert store.search(emb.embed_query("anything"), k=3) == []


def test_store_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore().add([_chunk("1", "x")], np.zeros((2, 4), dtype=np.float32))


def test_store_persists_to_disk(tmp_path: Path) -> None:
    emb = HashingEmbedder()
    store = InMemoryVectorStore(persist_dir=tmp_path)
    chunk = _chunk("1", "persist me")
    store.add([chunk], emb.embed_documents([chunk.text]))

    reloaded = build_store(Settings(index_dir=tmp_path))
    assert [c.id for c in reloaded.all_chunks()] == ["1"]
    assert reloaded.search(emb.embed_query("persist"), k=1)[0].chunk.id == "1"
