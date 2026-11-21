from __future__ import annotations

from rag_docs.embeddings import HashingEmbedder
from rag_docs.models import Chunk, ScoredChunk
from rag_docs.retrieval import (
    BM25Index,
    HybridRetriever,
    LexicalReranker,
    LLMReranker,
    NoopReranker,
    reciprocal_rank_fusion,
)
from rag_docs.stores import InMemoryVectorStore

TEXTS = [
    "Invoices are issued on the first business day of each month.",
    "Plan code ENT-42 includes single sign-on and audit logs.",
    "Customers can request a refund within 30 days of purchase.",
    "Our offices are located in Lisbon and Porto.",
]


def _chunks() -> list[Chunk]:
    return [
        Chunk(id=str(i), document_id="d", source="kb.pdf", page_number=i + 1, text=t)
        for i, t in enumerate(TEXTS)
    ]


def _hit(cid: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=cid, document_id="d", source="s", page_number=1, text=cid), score=score
    )


def test_bm25_finds_exact_identifiers() -> None:
    hits = BM25Index(_chunks()).search("What does ENT-42 include?", k=2)
    assert hits[0].chunk.id == "1"


def test_bm25_returns_nothing_for_unknown_terms() -> None:
    assert BM25Index(_chunks()).search("zebra", k=3) == []
    assert BM25Index([]).search("refund", k=3) == []


def test_rrf_rewards_agreement_between_rankers() -> None:
    fused = reciprocal_rank_fusion([[_hit("a"), _hit("b")], [_hit("b"), _hit("c")]])
    assert [h.chunk.id for h in fused] == ["b", "a", "c"]


def _retriever(mode: str) -> HybridRetriever:
    store = InMemoryVectorStore()
    emb = HashingEmbedder()
    chunks = _chunks()
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return HybridRetriever(store, emb, mode=mode)  # type: ignore[arg-type]


def test_each_retrieval_mode_finds_the_refund_chunk() -> None:
    for mode in ("dense", "bm25", "hybrid"):
        hits = _retriever(mode).retrieve("how long do I have to ask for a refund", k=2)
        assert hits[0].chunk.id == "2", mode


def test_noop_reranker_truncates() -> None:
    assert len(NoopReranker().rerank("q", [_hit("a"), _hit("b")], top_n=1)) == 1


def test_lexical_reranker_promotes_term_coverage() -> None:
    hits = [
        ScoredChunk(chunk=_chunks()[3], score=0.9),
        ScoredChunk(chunk=_chunks()[2], score=0.1),
    ]
    out = LexicalReranker().rerank("refund within 30 days", hits, top_n=2)
    assert out[0].chunk.id == "2"
    assert LexicalReranker().rerank("the a", hits, top_n=1)[0].chunk.id == "3"


class _ScriptedLLM:
    model = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return self.reply


def test_llm_reranker_orders_by_model_scores() -> None:
    hits = [_hit("a"), _hit("b"), _hit("c")]
    out = LLMReranker(_ScriptedLLM('Sure: {"scores": [2, 9, 5]}')).rerank("q", hits, top_n=2)
    assert [h.chunk.id for h in out] == ["b", "c"]
    assert out[0].score == 9.0


def test_llm_reranker_falls_back_on_bad_output() -> None:
    hits = [_hit("a"), _hit("b")]
    for reply in ("no json", '{"scores": [1]}', '{"other": 1}', "{bad", '{"scores": ["x", 1]}'):
        out = LLMReranker(_ScriptedLLM(reply)).rerank("q", hits, top_n=2)
        assert [h.chunk.id for h in out] == ["a", "b"], reply
    assert LLMReranker(_ScriptedLLM("")).rerank("q", [], top_n=2) == []
