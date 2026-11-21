from __future__ import annotations

from openai import OpenAI


class OpenAILLM:
    def __init__(self, model: str = "gpt-5.6-luna", client: OpenAI | None = None) -> None:
        self.model = model
        self._client = client or OpenAI()

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()
