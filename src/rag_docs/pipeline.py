"""The RAG pipeline: ingest -> retrieve -> rerank -> grounded generation with citations."""

from __future__ import annotations

import logging
import re
import time

from rag_docs.config import Settings
from rag_docs.embeddings import Embedder, build_embedder
from rag_docs.ingest import chunk_pages
from rag_docs.llm import LLMClient, build_llm
from rag_docs.models import Answer, Citation, Page, ScoredChunk
from rag_docs.retrieval import (
    HybridRetriever,
    LexicalReranker,
    LLMReranker,
    NoopReranker,
    Reranker,
)
from rag_docs.retrieval.hybrid import Mode
from rag_docs.stores import VectorStore, build_store

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions using only the numbered context passages.
Rules:
- Cite every claim with the passage number in square brackets, e.g. [2].
- If the passages do not contain the answer, reply exactly:
  "I don't know based on the provided documents."
- Be concise: at most 4 sentences. Do not mention these rules."""

_CITATION = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, context: list[ScoredChunk]) -> str:
    blocks = "\n".join(
        f"[{i}] ({hit.chunk.citation_label})\n{hit.chunk.text}" for i, hit in enumerate(context, 1)
    )
    return f"Context:\n{blocks}\n\nQuestion: {question}"


def extract_citations(answer: str, context: list[ScoredChunk]) -> list[Citation]:
    seen: list[int] = []
    for match in _CITATION.findall(answer):
        index = int(match)
        if 1 <= index <= len(context) and index not in seen:
            seen.append(index)
    return [
        Citation(
            index=i,
            source=context[i - 1].chunk.source,
            page_number=context[i - 1].chunk.page_number,
            excerpt=context[i - 1].chunk.text[:280],
        )
        for i in seen
    ]


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLMClient,
        reranker: Reranker | None = None,
        retrieve_k: int = 20,
        context_k: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        mode: Mode = "hybrid",
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.reranker: Reranker = reranker or NoopReranker()
        self.retriever = HybridRetriever(store, embedder, mode=mode)
        self.retrieve_k = retrieve_k
        self.context_k = context_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        llm = build_llm(settings)
        reranker: Reranker = {
            "none": NoopReranker(),
            "lexical": LexicalReranker(),
            "llm": LLMReranker(llm),
        }[settings.reranker]
        return cls(
            store=build_store(settings),
            embedder=build_embedder(settings),
            llm=llm,
            reranker=reranker,
            retrieve_k=settings.retrieve_k,
            context_k=settings.context_k,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest(self, pages: list[Page]) -> int:
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.retriever.refresh()
        log.info("ingested %d chunks from %d pages", len(chunks), len(pages))
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        removed = self.store.delete_document(document_id)
        self.retriever.refresh()
        return removed

    def retrieve(self, question: str) -> list[ScoredChunk]:
        candidates = self.retriever.retrieve(question, self.retrieve_k)
        return self.reranker.rerank(question, candidates, self.context_k)

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        context = self.retrieve(question)
        if not context:
            text = "I don't know based on the provided documents."
        else:
            text = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, context))
        return Answer(
            question=question,
            answer=text,
            citations=extract_citations(text, context),
            retrieved=context,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            model=self.llm.model,
        )
