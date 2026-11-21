"""Core data types shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Page(BaseModel):
    """One page of extracted text from a source document."""

    document_id: str
    source: str
    page_number: int = Field(ge=1)
    text: str


class Chunk(BaseModel):
    """A retrievable unit of text with enough metadata to cite it."""

    id: str
    document_id: str
    source: str
    page_number: int
    text: str

    @property
    def citation_label(self) -> str:
        return f"{self.source} p.{self.page_number}"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    index: int
    source: str
    page_number: int
    excerpt: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[ScoredChunk]
    latency_ms: float
    model: str
