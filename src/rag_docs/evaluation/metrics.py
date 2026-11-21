"""Deterministic metrics: cheap, reproducible, and safe to run in CI."""

from __future__ import annotations

from rag_docs.evaluation.dataset import EvalItem
from rag_docs.models import Answer, ScoredChunk

ABSTAIN_MARKER = "i don't know"


def gold_rank(item: EvalItem, hits: list[ScoredChunk]) -> int | None:
    """1-based rank of the first chunk from the labelled page, or None."""
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.source == item.source and hit.chunk.page_number == item.page:
            return rank
    return None


def hit_at_k(item: EvalItem, hits: list[ScoredChunk], k: int) -> float:
    rank = gold_rank(item, hits)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(item: EvalItem, hits: list[ScoredChunk]) -> float:
    rank = gold_rank(item, hits)
    return 1.0 / rank if rank else 0.0


def abstained(answer: Answer) -> bool:
    return ABSTAIN_MARKER in answer.answer.lower()


def answer_correct(item: EvalItem, answer: Answer) -> float:
    """Answerable: every key fact appears. Unanswerable: the model abstains."""
    if not item.answerable:
        return 1.0 if abstained(answer) else 0.0
    if abstained(answer):
        return 0.0
    text = answer.answer.lower()
    return 1.0 if all(fact.lower() in text for fact in item.must_include) else 0.0


def citation_correct(item: EvalItem, answer: Answer) -> float:
    """At least one citation points at the labelled page (answerable items only)."""
    return (
        1.0
        if any(c.source == item.source and c.page_number == item.page for c in answer.citations)
        else 0.0
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
