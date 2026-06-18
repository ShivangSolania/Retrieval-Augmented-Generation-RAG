"""
CLI Runner for the RAG Pipeline
================================

Run the RAG pipeline directly from the command line without starting
the FastAPI server.  Supports all core operations:

* ``ingest``       — Parse and store a document
# * ``query``        — Full RAG pipeline (retrieve + generate answer) DONT USE, USE SEARCH
* ``search``       — Retrieve and rerank chunks (no generation)
* ``health``       — Show collection stats
* ``diagnostics``  — Per-document chunk diagnostics
* ``interactive``  — Interactive Q&A loop

Usage::

    python -m llamaindex_rag.cli ingest --file report.pdf --doc-id doc-001
    python -m llamaindex_rag.cli search "What are the key findings?"
    python -m llamaindex_rag.cli interactive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Load .env from the package directory so OPENAI_API_KEY is available
# before any LlamaIndex / OpenAI imports that need it.
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

from pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Globals (lazy-initialised)
# ---------------------------------------------------------------------------

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """Return (and cache) a single pipeline instance."""
    global _pipeline
    if _pipeline is None:
        print("[*] Initialising RAG pipeline (first run downloads models, "
              "this may take a few minutes) ...")
        t0 = time.time()
        _pipeline = RAGPipeline()
        print(f"[*] Pipeline ready in {time.time() - t0:.1f}s\n")
    return _pipeline


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a document into the pipeline."""
    pipeline = _get_pipeline()
    print(f"[>] Ingesting: {args.file}")
    print(f"    doc_id:    {args.doc_id}\n")

    result = asyncio.run(pipeline.ingest(args.file, args.doc_id))

    print("=== Ingestion Result ===")
    print(f"  doc_id:            {result.doc_id}")
    print(f"  chunk_count:       {result.chunk_count}")
    print(f"  toc_entries:       {result.toc_entries}")
    print(f"  ingestion_time_ms: {result.ingestion_time_ms}")
    print()


def cmd_query(args: argparse.Namespace) -> None:
    """Run a full RAG query (retrieve + generate answer)."""
    pipeline = _get_pipeline()
    query_text = " ".join(args.query)
    print(f'[>] Query: "{query_text}"\n')

    result = asyncio.run(pipeline.query(query_text))

    print("=== Answer ===")
    print(result.answer)
    print()

    if result.citations:
        print("=== Citations ===")
        for c in result.citations:
            print(f"  [{c.index}] {c.section_path}  "
                  f"(page {c.page}, {c.chunk_type}, score={c.score:.4f})")
        print()

    print(f"  Latency: {result.latency_ms}ms")
    print()


def cmd_search(args: argparse.Namespace) -> None:
    """Search for relevant chunks without generating an answer."""
    pipeline = _get_pipeline()
    query_text = " ".join(args.query)
    print(f'[>] Searching: "{query_text}"  (top_k={args.top_k})\n')

    try:
        result = asyncio.run(
            pipeline.search(query_text, top_k=args.top_k)
        )
    except RuntimeError as exc:
        print(f"[!] {exc}")
        sys.exit(1)

    print(f"=== Search Results ({len(result.chunks)} chunks) ===\n")
    for i, chunk in enumerate(result.chunks, 1):
        meta = chunk.metadata
        print(f"--- [{i}]  score={chunk.score:.4f} ---")
        print(f"  section: {meta.get('section_path', 'N/A')}")
        print(f"  page:    {meta.get('page', '?')}")
        print(f"  type:    {meta.get('chunk_type', '?')}")
        text_preview = chunk.text[:200].replace("\n", " ")
        print(f"  text:    {text_preview}...")
        print()

    print(f"  Latency: {result.latency_ms}ms")
    print()


def cmd_health(args: argparse.Namespace) -> None:
    """Show collection statistics."""
    pipeline = _get_pipeline()
    pipeline._ensure_loaded_from_store()

    print("=== Health / Status ===")
    print(f"  status:          ok")
    print(f"  chunks_count:    {pipeline.store.chunks_collection.count()}")
    print(f"  toc_count:       {pipeline.store.toc_collection.count()}")
    print(f"  bm25_index_size: {len(pipeline.all_nodes)}")
    print()


def cmd_diagnostics(args: argparse.Namespace) -> None:
    """Show chunk diagnostics for a specific document."""
    pipeline = _get_pipeline()
    doc_id = args.doc_id

    nodes = [
        n for n in pipeline.all_nodes
        if n.metadata.get("doc_id") == doc_id
    ]

    if not nodes:
        print(f"[!] No chunks found for doc_id='{doc_id}'.")
        sys.exit(1)

    depths = [n.metadata.get("depth", 0) for n in nodes]
    depth_dist = dict(Counter(str(d) for d in depths))
    avg_importance = sum(
        n.metadata.get("importance_score", 0.0) for n in nodes
    ) / max(len(nodes), 1)

    print(f"=== Diagnostics for '{doc_id}' ===")
    print(f"  chunk_count:        {len(nodes)}")
    print(f"  depth_distribution: {json.dumps(depth_dist)}")
    print(f"  avg_importance:     {avg_importance:.4f}")
    print()


def cmd_interactive(args: argparse.Namespace) -> None:
    """Start an interactive Q&A session."""
    pipeline = _get_pipeline()

    if pipeline.chunk_index is None:
        print("[!] No documents ingested yet.")
        print("[*] You can ingest inline by typing:  /ingest <file_path> <doc_id>")
        print()

    print("=" * 60)
    print("  LlamaIndex RAG — Interactive Mode")
    print("  Type your question and press Enter.")
    print("  Commands:")
    print("    /ingest <file_path> <doc_id>  — Ingest a document")
    print("    /health                       — Show stats")
    print("    /quit  or  /exit              — Exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Goodbye!")
            break

        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.lower() in ("/quit", "/exit"):
            print("[*] Goodbye!")
            break

        if user_input.lower() == "/health":
            print(f"  chunks:    {pipeline.store.chunks_collection.count()}")
            print(f"  toc:       {pipeline.store.toc_collection.count()}")
            print(f"  bm25:      {len(pipeline.all_nodes)}")
            print()
            continue

        if user_input.lower().startswith("/ingest "):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("[!] Usage: /ingest <file_path> <doc_id>")
                continue
            _, fpath, did = parts
            print(f"[*] Ingesting {fpath} as '{did}' ...")
            result = asyncio.run(pipeline.ingest(fpath, did))
            print(f"[*] Done — {result.chunk_count} chunks, "
                  f"{result.toc_entries} TOC entries, "
                  f"{result.ingestion_time_ms}ms\n")
            continue

        # --- Normal query ---
        if pipeline.chunk_index is None:
            print("[!] No documents ingested. Use /ingest first.\n")
            continue

        print("[*] Thinking ...\n")
        result = asyncio.run(pipeline.query(user_input))

        print(f"Bot> {result.answer}\n")
        if result.citations:
            print("  Citations:")
            for c in result.citations:
                print(f"    [{c.index}] {c.section_path} "
                      f"(p.{c.page}, score={c.score:.3f})")
            print()
        print(f"  ({result.latency_ms}ms)\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m llamaindex_rag.cli",
        description="LlamaIndex RAG Pipeline — Command-Line Interface",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument(
        "--file", "-f", required=True,
        help="Absolute path to the document file",
    )
    p_ingest.add_argument(
        "--doc-id", "-d", required=True,
        help="Unique document identifier",
    )

    # --- query ---
    p_query = sub.add_parser("query", help="Full RAG query (retrieve + answer)")
    p_query.add_argument(
        "query", nargs="+",
        help="Your question (quote or space-separated)",
    )

    # --- search ---
    p_search = sub.add_parser("search", help="Search chunks (no LLM answer)")
    p_search.add_argument(
        "query", nargs="+",
        help="Search query",
    )
    p_search.add_argument(
        "--top-k", "-k", type=int, default=10,
        help="Number of results (default: 10)",
    )

    # --- health ---
    sub.add_parser("health", help="Show collection stats")

    # --- diagnostics ---
    p_diag = sub.add_parser("diagnostics", help="Document chunk diagnostics")
    p_diag.add_argument(
        "doc_id",
        help="Document ID to inspect",
    )

    # --- interactive ---
    sub.add_parser("interactive", help="Interactive Q&A session")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "search": cmd_search,
        "health": cmd_health,
        "diagnostics": cmd_diagnostics,
        "interactive": cmd_interactive,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
