from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_docs import cli
from rag_docs.api import create_app
from rag_docs.config import Settings, get_settings
from rag_docs.embeddings import HashingEmbedder
from rag_docs.llm import ExtractiveLLM
from rag_docs.pipeline import RAGPipeline
from rag_docs.retrieval import LexicalReranker
from rag_docs.stores import InMemoryVectorStore


@pytest.fixture
def client() -> TestClient:
    rag = RAGPipeline(
        store=InMemoryVectorStore(),
        embedder=HashingEmbedder(),
        llm=ExtractiveLLM(),
        reranker=LexicalReranker(),
        chunk_size=300,
        chunk_overlap=40,
    )
    app = create_app(Settings(llm_provider="extractive", embedding_provider="hashing"), rag)
    with TestClient(app) as c:
        yield c


def test_upload_query_list_delete(client: TestClient, two_page_pdf: bytes) -> None:
    res = client.post("/documents", files={"file": ("policy.pdf", two_page_pdf, "application/pdf")})
    assert res.status_code == 201
    doc = res.json()
    assert doc["pages"] == 2 and doc["chunks"] >= 2

    assert client.get("/health").json()["chunks"] == doc["chunks"]
    assert client.get("/documents").json()[0]["source"] == "policy.pdf"

    answer = client.post("/query", json={"question": "When is the support team available?"}).json()
    assert "Monday to Friday" in answer["answer"]
    assert answer["citations"][0]["page_number"] == 2

    assert client.delete(f"/documents/{doc['document_id']}").status_code == 204
    assert client.delete(f"/documents/{doc['document_id']}").status_code == 404


def test_upload_validation(client: TestClient) -> None:
    assert (
        client.post(
            "/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/documents", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")}
        ).status_code
        == 422
    )
    assert (
        client.post("/documents", files={"file": ("empty.md", b"   ", "text/markdown")}).status_code
        == 422
    )
    assert (
        client.post(
            "/documents",
            files={"file": ("notes.md", b"Lisbon office opens at 9.", "text/markdown")},
        ).status_code
        == 201
    )


def test_query_validation_and_ui(client: TestClient) -> None:
    assert client.post("/query", json={"question": "a"}).status_code == 422
    assert "rag-docs-assistant" in client.get("/").text


def test_cli_ingest_and_ask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    two_page_pdf: bytes,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.pdf").write_bytes(two_page_pdf)
    (docs / "ignore.png").write_bytes(b"x")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("RAG_RERANKER", "lexical")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    get_settings.cache_clear()
    try:
        assert cli.main(["ingest", str(docs)]) == 0
        assert "from 1 file" in capsys.readouterr().out

        assert cli.main(["ask", "Within how many days can customers request a refund?"]) == 0
        out = capsys.readouterr().out
        assert "30 days" in out and "policy.pdf, page 1" in out

        assert cli.main(["ask", "--json", "Who has a dedicated account manager?"]) == 0
        assert json.loads(capsys.readouterr().out)["citations"]
    finally:
        get_settings.cache_clear()
