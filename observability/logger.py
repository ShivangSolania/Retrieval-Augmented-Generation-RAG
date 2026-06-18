"""
Structured JSON Logger
======================

Provides :class:`RAGLogger` for structured, machine-parsable logging
across all pipeline stages (ingestion, retrieval, generation).

All output is JSON — no ``print()`` calls anywhere in the pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from llama_index.core.schema import NodeWithScore


def _setup_json_logger(name: str) -> logging.Logger:
    """Create a logger that emits single-line JSON to *stdout*.

    Args:
        name: Logger name (typically ``"rag_pipeline"``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class _JsonFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* into a JSON string.

        Args:
            record: The log record to format.

        Returns:
            Single-line JSON string.
        """
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        if hasattr(record, "data"):
            log_entry["data"] = record.data  # type: ignore[attr-defined]
        return json.dumps(log_entry, default=str)


class RAGLogger:
    """Structured logger for all RAG pipeline operations.

    Every method emits a single JSON log line capturing the event name
    and associated metrics / diagnostics.  This is the *only* logging
    mechanism used by the pipeline — no ``print()`` statements.
    """

    def __init__(self) -> None:
        self._logger = _setup_json_logger("rag_pipeline")

    def log_ingestion(
        self,
        doc_id: str,
        chunk_count: int,
        toc_count: int,
        latency_ms: int,
        latency_sec: float = 0.0,
    ) -> None:
        """Log a document ingestion event.

        Args:
            doc_id: Unique document identifier.
            chunk_count: Number of content chunks produced.
            toc_count: Number of TOC index entries produced.
            latency_ms: Wall-clock ingestion time in milliseconds.
            latency_sec: Wall-clock ingestion time in seconds.
        """
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg="ingestion_complete",
            args=(),
            exc_info=None,
        )
        record.data = {  # type: ignore[attr-defined]
            "doc_id": doc_id,
            "chunk_count": chunk_count,
            "toc_count": toc_count,
            "latency_ms": latency_ms,
            "latency_sec": latency_sec,
        }
        self._logger.handle(record)

    def log_retrieval(
        self,
        query: str,
        candidates: list[NodeWithScore],
        reranked: list[NodeWithScore],
        latency_ms: int,
        latency_sec: float = 0.0,
    ) -> None:
        """Log a retrieval event with full depth/section diagnostics.

        This is the primary mechanism for verifying that deep document
        content is being reached by the retrieval pipeline.

        Args:
            query: The user's original query string.
            candidates: All candidate nodes before reranking.
            reranked: Final nodes after cross-encoder reranking.
            latency_ms: Wall-clock retrieval time in milliseconds.
            latency_sec: Wall-clock retrieval time in seconds.
        """
        reranked_details = [
            {
                "rank": i,
                "node_id": n.node_id,
                "score": n.score,
                "section_path": n.metadata.get("section_path", ""),
                "depth": n.metadata.get("depth", 0),
                "page": n.metadata.get("page", 0),
                "chunk_type": n.metadata.get("chunk_type", ""),
            }
            for i, n in enumerate(reranked, 1)
        ]
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg="retrieval_complete",
            args=(),
            exc_info=None,
        )
        record.data = {  # type: ignore[attr-defined]
            "query": query,
            "candidate_count": len(candidates),
            "reranked_count": len(reranked),
            "latency_ms": latency_ms,
            "latency_sec": latency_sec,
            "reranked_chunks": reranked_details,
        }
        self._logger.handle(record)

    def log_generation(
        self,
        query: str,
        chunks_used: int,
        latency_ms: int,
        latency_sec: float = 0.0,
    ) -> None:
        """Log a generation (LLM synthesis) event.

        Args:
            query: The user's original query string.
            chunks_used: Number of context chunks sent to the LLM.
            latency_ms: Wall-clock generation time in milliseconds.
            latency_sec: Wall-clock generation time in seconds.
        """
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg="generation_complete",
            args=(),
            exc_info=None,
        )
        record.data = {  # type: ignore[attr-defined]
            "query": query,
            "chunks_used": chunks_used,
            "latency_ms": latency_ms,
            "latency_sec": latency_sec,
        }
        self._logger.handle(record)
