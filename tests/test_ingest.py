from __future__ import annotations

from pathlib import Path

import pytest

from rag_docs.ingest import chunk_pages, document_id_for, load_pdf, load_text
from rag_docs.ingest.loaders import load_path, normalise
from rag_docs.models import Page


def test_load_pdf_keeps_page_numbers(two_page_pdf: bytes) -> None:
    pages = load_pdf(two_page_pdf, "policy.pdf")
    assert [p.page_number for p in pages] == [1, 2]
    assert "30 days" in pages[0].text
    assert "Monday to Friday" in pages[1].text
    assert all(p.source == "policy.pdf" for p in pages)


def test_document_id_is_content_addressed(two_page_pdf: bytes) -> None:
    assert document_id_for(two_page_pdf) == document_id_for(two_page_pdf)
    assert document_id_for(b"a") != document_id_for(b"b")


def test_normalise_joins_hyphenated_line_breaks_and_collapses_space() -> None:
    assert normalise("retriev-\nal   augmented\n\n\n\ngeneration") == (
        "retrieval augmented\n\ngeneration"
    )


def test_load_text_skips_empty_files() -> None:
    assert load_text(b"   \n ", "empty.md") == []
    pages = load_text(b"# Title\n\nBody", "notes.md")
    assert pages[0].page_number == 1


def test_load_path_dispatches_on_suffix(tmp_path: Path, two_page_pdf: bytes) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(two_page_pdf)
    txt = tmp_path / "b.txt"
    txt.write_text("hello world")
    assert len(load_path(pdf)) == 2
    assert load_path(txt)[0].text == "hello world"


def _page(text: str) -> Page:
    return Page(document_id="d", source="s.pdf", page_number=3, text=text)


def test_short_page_is_one_chunk() -> None:
    chunks = chunk_pages([_page("Short text.")], chunk_size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].id == "d:3:0"
    assert chunks[0].citation_label == "s.pdf p.3"


def test_chunks_respect_size_and_overlap() -> None:
    sentences = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk_pages([_page(sentences)], chunk_size=200, overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    # Neighbouring chunks share some text, so boundary facts stay retrievable.
    first_words = set(chunks[0].text.split()[-4:])
    assert first_words & set(chunks[1].text.split()[:12])


def test_chunking_prefers_paragraph_boundaries() -> None:
    text = "A" * 90 + ".\n\n" + "B" * 90 + "."
    chunks = chunk_pages([_page(text)], chunk_size=120, overlap=0)
    assert [c.text[0] for c in chunks] == ["A", "B"]


def test_overlong_token_is_hard_cut() -> None:
    chunks = chunk_pages([_page("x" * 450)], chunk_size=200, overlap=0)
    assert [len(c.text) for c in chunks] == [200, 200, 50]


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_pages([_page("text")], chunk_size=100, overlap=100)
