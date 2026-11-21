from __future__ import annotations

from typing import TYPE_CHECKING

from rag_docs.llm.base import LLMClient
from rag_docs.llm.extractive import ExtractiveLLM

if TYPE_CHECKING:
    from rag_docs.config import Settings

__all__ = ["ExtractiveLLM", "LLMClient", "build_llm"]


def build_llm(settings: Settings) -> LLMClient:
    model = settings.resolved_llm_model()
    if settings.llm_provider == "anthropic":
        from rag_docs.llm.anthropic import AnthropicLLM

        return AnthropicLLM(model=model)
    if settings.llm_provider == "openai":
        from rag_docs.llm.openai import OpenAILLM

        return OpenAILLM(model=model)
    return ExtractiveLLM()
