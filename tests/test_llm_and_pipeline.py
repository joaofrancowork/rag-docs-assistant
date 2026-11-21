from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_docs.config import Settings
from rag_docs.embeddings import HashingEmbedder
from rag_docs.ingest import load_pdf
from rag_docs.llm import ExtractiveLLM, build_llm
from rag_docs.llm.anthropic import AnthropicLLM
from rag_docs.llm.openai import OpenAILLM
from rag_docs.pipeline import RAGPipeline, build_prompt, extract_citations
from rag_docs.retrieval import LexicalReranker
from rag_docs.stores import InMemoryVectorStore


def _pipeline(**kwargs: object) -> RAGPipeline:
    return RAGPipeline(
        store=InMemoryVectorStore(),
        embedder=HashingEmbedder(),
        llm=ExtractiveLLM(),
        reranker=LexicalReranker(),
        chunk_size=300,
        chunk_overlap=40,
        **kwargs,  # type: ignore[arg-type]
    )


def test_end_to_end_answer_is_grounded_and_cited(two_page_pdf: bytes) -> None:
    rag = _pipeline()
    assert rag.ingest(load_pdf(two_page_pdf, "policy.pdf")) > 0
    answer = rag.ask("Within how many days can customers request a refund?")
    assert "30 days" in answer.answer
    assert answer.citations[0].source == "policy.pdf"
    assert answer.citations[0].page_number == 1
    assert answer.model == "extractive-baseline"


def test_unanswerable_question_abstains(two_page_pdf: bytes) -> None:
    rag = _pipeline()
    rag.ingest(load_pdf(two_page_pdf, "policy.pdf"))
    answer = rag.ask("What is the CEO's favourite colour?")
    assert answer.answer.startswith("I don't know")
    assert answer.citations == []


def test_empty_index_abstains_without_calling_the_model() -> None:
    llm = MagicMock()
    llm.model = "mock"
    rag = RAGPipeline(store=InMemoryVectorStore(), embedder=HashingEmbedder(), llm=llm)
    assert rag.ask("anything").answer.startswith("I don't know")
    llm.complete.assert_not_called()
    assert rag.ingest([]) == 0


def test_delete_document_removes_it_from_retrieval(two_page_pdf: bytes) -> None:
    rag = _pipeline()
    pages = load_pdf(two_page_pdf, "policy.pdf")
    rag.ingest(pages)
    assert rag.delete_document(pages[0].document_id) > 0
    assert rag.retrieve("refund") == []


def test_extract_citations_ignores_out_of_range_and_duplicates(two_page_pdf: bytes) -> None:
    rag = _pipeline()
    rag.ingest(load_pdf(two_page_pdf, "policy.pdf"))
    context = rag.retrieve("refund policy support hours")
    citations = extract_citations("A [1]. B [1]. C [99].", context)
    assert [c.index for c in citations] == [1]
    assert "[1] (policy.pdf p." in build_prompt("q", context)


def test_from_settings_wires_every_reranker(tmp_path: object) -> None:
    for reranker in ("none", "lexical", "llm"):
        settings = Settings(
            llm_provider="extractive",
            embedding_provider="hashing",
            reranker=reranker,  # type: ignore[arg-type]
            index_dir=tmp_path,  # type: ignore[arg-type]
        )
        assert RAGPipeline.from_settings(settings).llm.model == "extractive-baseline"


def test_resolved_model_defaults() -> None:
    assert Settings(llm_provider="anthropic").resolved_llm_model() == "claude-opus-5-5"
    assert Settings(llm_provider="openai", llm_model="x").resolved_llm_model() == "x"


def test_anthropic_client_sends_system_and_joins_text() -> None:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", text=""),
            SimpleNamespace(type="text", text=" Answer [1] "),
        ],
    )
    llm = AnthropicLLM(client=client)
    assert llm.complete("sys", "prompt") == "Answer [1]"
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == "sys"
    assert kwargs["output_config"] == {"effort": "low"}


def test_anthropic_client_handles_refusal() -> None:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(stop_reason="refusal", content=[])
    assert "can't help" in AnthropicLLM(client=client).complete("s", "p")


def test_openai_client_maps_messages() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" hi "))]
    )
    assert OpenAILLM(client=client).complete("sys", "p") == "hi"
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}


def test_build_llm_providers(monkeypatch: object) -> None:
    import rag_docs.llm.anthropic as anth
    import rag_docs.llm.openai as oai

    monkeypatch.setattr(anth.anthropic, "Anthropic", MagicMock())  # type: ignore[attr-defined]
    monkeypatch.setattr(oai, "OpenAI", MagicMock())  # type: ignore[attr-defined]
    assert isinstance(build_llm(Settings(llm_provider="anthropic")), AnthropicLLM)
    assert isinstance(build_llm(Settings(llm_provider="openai")), OpenAILLM)
    assert isinstance(build_llm(Settings(llm_provider="extractive")), ExtractiveLLM)
