from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """The only surface the pipeline needs from a model provider.

    Keeping it this small is what makes providers interchangeable: switching
    from Claude to OpenAI is a config change, not a code change.
    """

    model: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...
