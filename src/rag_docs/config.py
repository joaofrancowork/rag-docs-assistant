"""Runtime configuration, read from environment variables prefixed with ``RAG_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai", "extractive"]
EmbeddingProvider = Literal["openai", "fastembed", "hashing"]
VectorStoreKind = Literal["memory", "chroma"]
RerankerKind = Literal["none", "lexical", "llm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    llm_provider: LLMProvider = "anthropic"
    llm_model: str | None = None
    embedding_provider: EmbeddingProvider = "fastembed"
    embedding_model: str | None = None
    vector_store: VectorStoreKind = "memory"
    chroma_url: str = "http://localhost:8001"
    reranker: RerankerKind = "llm"

    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    retrieve_k: int = Field(default=20, ge=1)
    context_k: int = Field(default=5, ge=1)
    index_dir: Path = Path(".index")

    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "anthropic": "claude-opus-5-5",
            "openai": "gpt-5.6-luna",
            "extractive": "extractive-baseline",
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
