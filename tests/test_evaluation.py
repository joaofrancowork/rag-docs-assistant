from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_docs import cli
from rag_docs.config import Settings, get_settings
from rag_docs.embeddings import HashingEmbedder
from rag_docs.evaluation import metrics
from rag_docs.evaluation.dataset import EvalItem, load_dataset
from rag_docs.evaluation.judge import Judge
from rag_docs.evaluation.runner import run_suite, to_markdown
from rag_docs.llm import ExtractiveLLM
from rag_docs.models import Answer, Chunk, Citation, ScoredChunk

from .conftest import make_pdf

ITEM = EvalItem(id="q", question="refund?", source="a.pdf", page=2, must_include=["30 days"])
NO_ANSWER = EvalItem(id="n", question="ceo?", source=None, page=None, must_include=[])


def _hit(source: str, page: int) -> ScoredChunk:
    chunk = Chunk(id=f"{source}{page}", document_id="d", source=source, page_number=page, text="t")
    return ScoredChunk(chunk=chunk, score=1.0)


def _answer(text: str, citations: list[Citation] | None = None) -> Answer:
    return Answer(
        question="refund?",
        answer=text,
        citations=citations or [],
        retrieved=[_hit("a.pdf", 2)],
        latency_ms=1.0,
        model="m",
    )


def test_retrieval_metrics() -> None:
    hits = [_hit("a.pdf", 1), _hit("a.pdf", 2)]
    assert metrics.gold_rank(ITEM, hits) == 2
    assert metrics.hit_at_k(ITEM, hits, 1) == 0.0
    assert metrics.hit_at_k(ITEM, hits, 5) == 1.0
    assert metrics.reciprocal_rank(ITEM, hits) == 0.5
    assert metrics.reciprocal_rank(ITEM, []) == 0.0
    assert metrics.mean([]) == 0.0


def test_answer_and_citation_metrics() -> None:
    good = _answer(
        "Within 30 days [1].", [Citation(index=1, source="a.pdf", page_number=2, excerpt="")]
    )
    assert metrics.answer_correct(ITEM, good) == 1.0
    assert metrics.citation_correct(ITEM, good) == 1.0
    assert metrics.answer_correct(ITEM, _answer("Within 14 days.")) == 0.0
    idk = _answer("I don't know based on the provided documents.")
    assert metrics.answer_correct(ITEM, idk) == 0.0
    assert metrics.answer_correct(NO_ANSWER, idk) == 1.0
    assert metrics.answer_correct(NO_ANSWER, good) == 0.0
    assert metrics.citation_correct(ITEM, idk) == 0.0


def test_load_dataset(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(ITEM.model_dump_json() + "\n\n" + NO_ANSWER.model_dump_json() + "\n")
    items = load_dataset(path)
    assert [i.answerable for i in items] == [True, False]
    path.write_text("{not json}\n")
    with pytest.raises(ValueError, match=r"q\.jsonl:1"):
        load_dataset(path)


def test_repository_dataset_is_valid() -> None:
    items = load_dataset(Path(__file__).parents[1] / "data/eval/questions.jsonl")
    assert len(items) >= 30
    assert len({i.id for i in items}) == len(items)
    assert any(not i.answerable for i in items)


class _Scripted:
    model = "judge"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return self.replies.pop(0)


def test_judge_faithfulness_and_relevance() -> None:
    reply = json.dumps(
        {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]}
    )
    judge = Judge(_Scripted(reply, '{"score": 5, "reason": "direct"}'))
    assert judge.faithfulness(_answer("x")) == 0.5
    assert judge.relevance(_answer("x")) == 1.0


def test_judge_handles_abstentions_and_bad_replies() -> None:
    judge = Judge(
        _Scripted('{"claims": []}', "nonsense", "{broken", '{"claims": 3}', '{"score": 9}', "[]")
    )
    assert judge.faithfulness(_answer("I don't know")) == 1.0
    assert judge.faithfulness(_answer("x")) is None
    assert judge.faithfulness(_answer("x")) is None
    assert judge.faithfulness(_answer("x")) is None
    assert judge.relevance(_answer("x")) is None
    assert judge.relevance(_answer("x")) is None


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(
        make_pdf(["Our offices are in Lisbon.", "Customers can request a refund within 30 days."])
    )
    return docs


def test_run_suite_compares_configurations(tmp_path: Path) -> None:
    llm = ExtractiveLLM()
    results = run_suite(
        [ITEM, NO_ANSWER],
        _docs(tmp_path),
        HashingEmbedder(),
        llm,
        Judge(llm),
        Settings(chunk_size=200, chunk_overlap=20),
    )
    assert [r.name for r in results][:3] == ["dense", "bm25", "hybrid"]
    hybrid = results[2]
    assert hybrid.hit_at_1 == 1.0 and hybrid.answer_accuracy == 1.0
    assert hybrid.abstention_accuracy == 1.0
    report = to_markdown(results, "hashing", "extractive", 2)
    assert "| hybrid |" in report and "Faithfulness" in report


def test_eval_cli_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "q.jsonl"
    dataset.write_text(ITEM.model_dump_json() + "\n")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hashing")
    get_settings.cache_clear()
    try:
        out = tmp_path / "reports"
        assert (
            cli.main(
                [
                    "eval",
                    "--dataset",
                    str(dataset),
                    "--docs",
                    str(_docs(tmp_path)),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        assert "Evaluation report" in (out / "eval_report.md").read_text()
        assert json.loads((out / "eval_results.json").read_text())[0]["name"] == "dense"
    finally:
        get_settings.cache_clear()
