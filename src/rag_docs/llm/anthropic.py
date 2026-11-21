from __future__ import annotations

from typing import Literal

import anthropic
from anthropic.types import MessageParam

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AnthropicLLM:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        client: anthropic.Anthropic | None = None,
        effort: Effort = "low",
    ) -> None:
        self.model = model
        self._client = client or anthropic.Anthropic()
        # Grounded QA over a short context is not a hard reasoning task: low effort
        # keeps latency and cost down without hurting answer quality.
        self._effort = effort

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self._effort},
        )
        if response.stop_reason == "refusal":
            return "I can't help with that request."
        return "".join(block.text for block in response.content if block.type == "text").strip()
