"""HTTP API and a minimal web UI."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_docs.config import Settings, get_settings
from rag_docs.ingest.loaders import load_pdf, load_text
from rag_docs.models import Answer
from rag_docs.pipeline import RAGPipeline

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class IngestResponse(BaseModel):
    document_id: str
    source: str
    pages: int
    chunks: int


class DocumentInfo(BaseModel):
    document_id: str
    source: str
    chunks: int


def get_pipeline(request: Request) -> RAGPipeline:
    rag: RAGPipeline = request.app.state.pipeline
    return rag


Rag = Annotated[RAGPipeline, Depends(get_pipeline)]


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or RAGPipeline.from_settings(settings)
        yield

    app = FastAPI(
        title="rag-docs-assistant",
        version="0.1.0",
        description="Ask questions about your PDFs and get cited answers.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            log.info(
                "request_id=%s method=%s path=%s status=%d latency_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                status,
                (time.perf_counter() - started) * 1000,
            )

    @app.get("/health")
    def health(rag: Rag) -> dict[str, str | int]:
        return {
            "status": "ok",
            "model": rag.llm.model,
            "embedder": rag.embedder.name,
            "chunks": len(rag.store.all_chunks()),
        }

    @app.post("/documents", status_code=201)
    async def upload(file: UploadFile, rag: Rag) -> IngestResponse:
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(415, f"Unsupported file type {suffix!r}; use PDF, TXT or MD")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 20 MB)")
        try:
            pages = load_pdf(content, name) if suffix == ".pdf" else load_text(content, name)
        except Exception as exc:  # pypdf raises a variety of errors on malformed files
            log.warning("could not parse %s: %s", name, exc)
            raise HTTPException(422, "Could not read this file") from exc
        if not pages:
            raise HTTPException(422, "No extractable text (scanned PDFs need OCR first)")
        chunks = rag.ingest(pages)
        return IngestResponse(
            document_id=pages[0].document_id, source=name, pages=len(pages), chunks=chunks
        )

    @app.get("/documents")
    def list_documents(rag: Rag) -> list[DocumentInfo]:
        docs: dict[str, DocumentInfo] = {}
        for chunk in rag.store.all_chunks():
            info = docs.setdefault(
                chunk.document_id,
                DocumentInfo(document_id=chunk.document_id, source=chunk.source, chunks=0),
            )
            info.chunks += 1
        return sorted(docs.values(), key=lambda d: d.source)

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: str, rag: Rag) -> None:
        if rag.delete_document(document_id) == 0:
            raise HTTPException(404, "Unknown document")

    @app.post("/query")
    def query(body: QueryRequest, rag: Rag) -> Answer:
        return rag.ask(body.question)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
