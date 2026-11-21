"""LLM-as-judge metrics for what string matching cannot see.

* Faithfulness: the share of claims in the answer supported by the retrieved
  context (catches hallucinations even when the answer "sounds right").
* Answer relevance: does the answer address the question (1-5, rescaled to 0-1)?

The judge gets a rubric and must return JSON; unusable replies are recorded as
missing instead of silently counting as a pass or a fail.
"""

from __future__ import annotations

import json
import re

from rag_docs.llm.base import LLMClient
from rag_docs.models import Answer

JUDGE_SYSTEM = "You are a strict evaluator of question-answering systems. Reply with JSON only."

FAITHFULNESS_PROMPT = """Split the ANSWER into atomic factual claims. For each claim, decide if
the CONTEXT explicitly supports it. Ignore citation markers like [1].
If the answer says it does not know, return {{"claims": []}}.
Return: {{"claims": [{{"claim": "...", "supported": true|false}}]}}

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEVANCE_PROMPT = """Rate how well the ANSWER addresses the QUESTION, ignoring whether it is
true. 5 = fully and directly answers it, 3 = partially, 1 = does not address it.
An honest "I don't know" scores 3.
Return: {{"score": <1-5>, "reason": "<one sentence>"}}

QUESTION: {question}
ANSWER: {answer}"""


def _json(reply: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def faithfulness(self, answer: Answer) -> float | None:
        context = "\n\n".join(h.chunk.text for h in answer.retrieved)
        reply = self.llm.complete(
            JUDGE_SYSTEM,
            FAITHFULNESS_PROMPT.format(context=context, answer=answer.answer),
            max_tokens=1024,
        )
        data = _json(reply)
        raw_claims = data.get("claims") if data else None
        if not isinstance(raw_claims, list):
            return None
        claims = [c for c in raw_claims if isinstance(c, dict)]
        if not claims:
            return 1.0  # an abstention makes no unsupported claims
        return sum(1 for c in claims if c.get("supported") is True) / len(claims)

    def relevance(self, answer: Answer) -> float | None:
        reply = self.llm.complete(
            JUDGE_SYSTEM,
            RELEVANCE_PROMPT.format(question=answer.question, answer=answer.answer),
            max_tokens=256,
        )
        data = _json(reply)
        score = data.get("score") if data else None
        if not isinstance(score, int | float) or not 1 <= score <= 5:
            return None
        return (float(score) - 1) / 4
