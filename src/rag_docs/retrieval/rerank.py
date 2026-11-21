"""Second-stage rerankers: cheap recall first, precise ordering second."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from rag_docs.llm.base import LLMClient
from rag_docs.models import ScoredChunk
from rag_docs.retrieval.bm25 import tokenize

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...


class NoopReranker:
    def rerank(self, query: str, hits: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return hits[:top_n]


class LexicalReranker:
    """Query-term coverage, with the first-stage score as a tie-breaker."""

    def rerank(self, query: str, hits: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        terms = set(tokenize(query))
        if not terms:
            return hits[:top_n]

        def coverage(hit: ScoredChunk) -> float:
            return len(terms & set(tokenize(hit.chunk.text))) / len(terms)

        rescored = [ScoredChunk(chunk=h.chunk, score=coverage(h)) for h in hits]
        order = sorted(
            range(len(hits)), key=lambda i: (rescored[i].score, hits[i].score), reverse=True
        )
        return [rescored[i] for i in order[:top_n]]


_RERANK_PROMPT = """You are ranking passages for a retrieval system.
Score how useful each passage is for answering the question, from 0 (irrelevant)
to 10 (directly answers it). Reply with JSON only: {{"scores": [<int per passage>]}}

Question: {question}

{passages}"""


class LLMReranker:
    """Listwise LLM scoring. Falls back to the first-stage order if the reply is unusable."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rerank(self, query: str, hits: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not hits:
            return []
        passages = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits))
        reply = self.llm.complete(
            system="You are a precise relevance judge.",
            prompt=_RERANK_PROMPT.format(question=query, passages=passages),
            max_tokens=512,
        )
        scores = _parse_scores(reply, len(hits))
        if scores is None:
            log.warning("LLM reranker returned an unusable reply; keeping first-stage order")
            return hits[:top_n]
        order = sorted(range(len(hits)), key=lambda i: (scores[i], -i), reverse=True)
        return [ScoredChunk(chunk=hits[i].chunk, score=float(scores[i])) for i in order[:top_n]]


def _parse_scores(reply: str, expected: int) -> list[float] | None:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))["scores"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(scores, list) or len(scores) != expected:
        return None
    try:
        return [float(s) for s in scores]
    except (TypeError, ValueError):
        return None
