"""Command line: ingest files, ask questions, run the evaluation, serve the API."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag_docs.config import get_settings
from rag_docs.ingest.loaders import load_path
from rag_docs.pipeline import RAGPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-docs")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index PDF, TXT or MD files")
    ingest.add_argument("paths", nargs="+", type=Path)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Print the full answer object")

    evaluate = sub.add_parser("eval", help="Run the evaluation suite")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/eval/questions.jsonl"))
    evaluate.add_argument("--docs", type=Path, default=Path("data/samples"))
    evaluate.add_argument("--out", type=Path, default=Path("reports"))
    evaluate.add_argument("--judge", action="store_true", help="Use the LLM as a judge too")

    serve = sub.add_parser("serve", help="Run the API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    if args.command == "serve":  # pragma: no cover - thin wrapper around uvicorn
        import uvicorn

        from rag_docs.api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0

    if args.command == "eval":
        from rag_docs.evaluation.runner import run_from_cli

        return run_from_cli(args.dataset, args.docs, args.out, judge=args.judge)

    rag = RAGPipeline.from_settings(settings)
    if args.command == "ingest":
        files = [
            p for path in args.paths for p in (sorted(path.rglob("*")) if path.is_dir() else [path])
        ]
        files = [f for f in files if f.suffix.lower() in {".pdf", ".txt", ".md"}]
        total = sum(rag.ingest(load_path(f)) for f in files)
        print(f"Indexed {total} chunks from {len(files)} file(s) into {settings.index_dir}")
        return 0

    answer = rag.ask(args.question)
    if args.json:
        print(json.dumps(answer.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(answer.answer)
        for c in answer.citations:
            print(f"  [{c.index}] {c.source}, page {c.page_number}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
