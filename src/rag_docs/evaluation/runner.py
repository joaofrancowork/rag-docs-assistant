"""Run the eval suite across retrieval configurations and write a report."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag_docs.config import Settings, get_settings
from rag_docs.embeddings import Embedder, build_embedder
from rag_docs.evaluation import metrics
from rag_docs.evaluation.dataset import EvalItem, load_dataset
from rag_docs.evaluation.judge import Judge
from rag_docs.ingest.loaders import load_path
from rag_docs.llm import LLMClient, build_llm
from rag_docs.pipeline import RAGPipeline
from rag_docs.retrieval import LexicalReranker, LLMReranker, NoopReranker, Reranker
from rag_docs.retrieval.hybrid import Mode
from rag_docs.stores import InMemoryVectorStore

log = logging.getLogger(__name__)


@dataclass
class ConfigResult:
    name: str
    hit_at_1: float
    hit_at_5: float
    mrr: float
    answer_accuracy: float
    citation_accuracy: float
    abstention_accuracy: float
    faithfulness: float | None = None
    relevance: float | None = None
    mean_latency_ms: float = 0.0
    failures: list[str] = field(default_factory=list)


def evaluate(
    name: str,
    rag: RAGPipeline,
    items: list[EvalItem],
    judge: Judge | None = None,
) -> ConfigResult:
    hit1, hit5, rr, correct, cited, abstain, latency = [], [], [], [], [], [], []
    faith: list[float] = []
    relevance: list[float] = []
    failures: list[str] = []
    for item in items:
        answer = rag.ask(item.question)
        latency.append(answer.latency_ms)
        ok = metrics.answer_correct(item, answer)
        correct.append(ok)
        if not ok:
            failures.append(item.id)
        if item.answerable:
            # Score retrieval on the final context the generator actually sees.
            hits = rag.retrieve(item.question)
            hit1.append(metrics.hit_at_k(item, hits, 1))
            hit5.append(metrics.hit_at_k(item, hits, 5))
            rr.append(metrics.reciprocal_rank(item, hits))
            cited.append(metrics.citation_correct(item, answer))
        else:
            abstain.append(ok)
        if judge:
            for scores, value in (
                (faith, judge.faithfulness(answer)),
                (relevance, judge.relevance(answer)),
            ):
                if value is not None:
                    scores.append(value)
    return ConfigResult(
        name=name,
        hit_at_1=metrics.mean(hit1),
        hit_at_5=metrics.mean(hit5),
        mrr=metrics.mean(rr),
        answer_accuracy=metrics.mean(correct),
        citation_accuracy=metrics.mean(cited),
        abstention_accuracy=metrics.mean(abstain),
        faithfulness=metrics.mean(faith) if faith else None,
        relevance=metrics.mean(relevance) if relevance else None,
        mean_latency_ms=round(metrics.mean(latency), 1),
        failures=failures,
    )


CONFIGS: list[tuple[str, Mode, str]] = [
    ("dense", "dense", "none"),
    ("bm25", "bm25", "none"),
    ("hybrid", "hybrid", "none"),
    ("hybrid + lexical rerank", "hybrid", "lexical"),
    ("hybrid + LLM rerank", "hybrid", "llm"),
]


def run_suite(
    items: list[EvalItem],
    docs_dir: Path,
    embedder: Embedder,
    llm: LLMClient,
    judge: Judge | None = None,
    settings: Settings | None = None,
) -> list[ConfigResult]:
    settings = settings or Settings()
    pages = [p for f in sorted(docs_dir.glob("*.pdf")) for p in load_path(f)]
    results = []
    for name, mode, reranker_kind in CONFIGS:
        rerankers: dict[str, Reranker] = {
            "none": NoopReranker(),
            "lexical": LexicalReranker(),
            "llm": LLMReranker(llm),
        }
        rag = RAGPipeline(
            store=InMemoryVectorStore(),
            embedder=embedder,
            llm=llm,
            reranker=rerankers[reranker_kind],
            retrieve_k=settings.retrieve_k,
            context_k=settings.context_k,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            mode=mode,
        )
        rag.ingest(pages)
        results.append(evaluate(name, rag, items, judge))
        log.info("evaluated %s", name)
    return results


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def to_markdown(results: list[ConfigResult], embedder: str, llm: str, n: int) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"{n} questions (including unanswerable ones) · embedder `{embedder}` · generator `{llm}`",
        "",
        "| Configuration | Hit@1 | Hit@5 | MRR | Answer acc. | Citation acc. | Abstention | "
        "Faithfulness | Relevance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {_pct(r.hit_at_1)} | {_pct(r.hit_at_5)} | {r.mrr:.2f} | "
            f"{_pct(r.answer_accuracy)} | {_pct(r.citation_accuracy)} | "
            f"{_pct(r.abstention_accuracy)} | {_pct(r.faithfulness)} | {_pct(r.relevance)} |"
        )
    return "\n".join(lines) + "\n"


def run_from_cli(dataset: Path, docs: Path, out: Path, judge: bool = False) -> int:
    settings = get_settings()
    items = load_dataset(dataset)
    embedder = build_embedder(settings)
    llm = build_llm(settings)
    results = run_suite(items, docs, embedder, llm, Judge(llm) if judge else None, settings)
    out.mkdir(parents=True, exist_ok=True)
    report = to_markdown(results, embedder.name, llm.model, len(items))
    (out / "eval_report.md").write_text(report, encoding="utf-8")
    (out / "eval_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print(report)
    return 0
