"""An offline 'LLM' that answers by extracting the best-matching sentence.

Two jobs: it lets the whole app (API, UI, evals) run with no API key, and it
is the baseline every real model has to beat in the evaluation report.
"""

from __future__ import annotations

import re

from rag_docs.retrieval.bm25 import tokenize

_CONTEXT_BLOCK = re.compile(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\Z)", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveLLM:
    model = "extractive-baseline"

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        if '"scores"' in prompt:  # used as a reranker: score by term overlap
            return self._score_passages(prompt)
        context, _, question = prompt.rpartition("\n\nQuestion:")
        terms = set(tokenize(question))
        best: tuple[float, str, str] = (0.0, "", "")
        for index, body in _CONTEXT_BLOCK.findall(context):
            for sentence in _SENTENCE.split(body.strip()):
                overlap = len(terms & set(tokenize(sentence))) / (len(terms) or 1)
                if overlap > best[0]:
                    best = (overlap, sentence.strip(), index)
        if best[0] < 0.25:
            return "I don't know based on the provided documents."
        return f"{best[1]} [{best[2]}]"

    def _score_passages(self, prompt: str) -> str:
        question = prompt.split("Question:", 1)[1].split("\n", 1)[0]
        terms = set(tokenize(question))
        passages = re.findall(r"^\[\d+\] (.*)$", prompt, re.MULTILINE)
        scores = [round(10 * len(terms & set(tokenize(p))) / (len(terms) or 1)) for p in passages]
        return '{"scores": ' + str(scores) + "}"
