"""
Response Generator
==================

Uses LlamaIndex's response synthesiser (``compact`` mode) to generate
answers from retrieved context chunks.  Produces structured output
including the answer text, per-chunk citations, and a full retrieval
trace for observability.

Also defines the shared Pydantic result models used across the pipeline:
:class:`Citation`, :class:`GenerationResult`, :class:`IngestResult`,
and :class:`QueryResult`.
"""

from __future__ import annotations

from typing import Any

from llama_index.core import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

from config import config


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single source citation from the retrieval trace.

    Attributes:
        index: 1-based rank in the context window.
        section_path: Hierarchical section path of the source chunk.
        page: Page number in the original document.
        chunk_type: Content type (heading, paragraph, table, …).
        score: Retrieval/reranking score.
    """

    index: int
    section_path: str = ""
    page: int = 0
    chunk_type: str = ""
    score: float = 0.0


class GenerationResult(BaseModel):
    """Complete generation output with answer, citations, and trace.

    Attributes:
        answer: The generated answer text.
        citations: Per-chunk source citations.
        retrieval_trace: Ordered list of trace dicts for debugging.
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = Field(default_factory=list)


class IngestResult(BaseModel):
    """Result of a document ingestion operation.

    Attributes:
        doc_id: Unique document identifier.
        chunk_count: Number of content chunks produced.
        toc_entries: Number of TOC index entries produced.
        ingestion_time_ms: Wall-clock ingestion time in milliseconds.
    """

    doc_id: str
    chunk_count: int
    toc_entries: int
    ingestion_time_ms: int
    ingestion_time_sec: float = 0.0


class QueryResult(BaseModel):
    """Complete query result including answer, citations, trace, and latency.

    Attributes:
        answer: The generated answer text.
        citations: Per-chunk source citations.
        trace: Ordered retrieval trace for debugging.
        latency_ms: End-to-end query latency in milliseconds.
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0
    latency_sec: float = 0.0


class RankedChunk(BaseModel):
    """A single ranked chunk from retrieval (no LLM generation).

    Attributes:
        text: Chunk text content.
        score: Reranker or retrieval score.
        metadata: Chunk metadata from ingestion.
        node_id: LlamaIndex node identifier.
    """

    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    node_id: str | None = None


class SearchResult(BaseModel):
    """Retrieve + rerank only — top-k chunks without LLM answer.

    Attributes:
        chunks: Ranked chunks after hybrid retrieval and reranking.
        trace: Ordered retrieval trace for debugging.
        latency_ms: End-to-end search latency in milliseconds.
    """

    chunks: list[RankedChunk] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0
    latency_sec: float = 0.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def get_llm() -> OpenAI:
    """Create an OpenAI LLM instance from the global config.

    Returns:
        Configured :class:`OpenAI` LLM.
    """
    return OpenAI(model=config.llm_model, temperature=0.1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """Citation-aware answer generator using LlamaIndex response synthesis.

    Uses ``compact`` response mode which concatenates all context chunks
    into a single prompt, minimising LLM calls while preserving full
    context.

    Usage::

        gen = Generator()
        result = await gen.generate("What is X?", ranked_nodes)
    """

    def __init__(self) -> None:
        self.synthesizer = get_response_synthesizer(
            llm=get_llm(),
            response_mode="compact",
            use_async=True,
        )

    async def generate(
        self,
        query: str,
        nodes: list[NodeWithScore],
    ) -> GenerationResult:
        """Generate an answer from *query* using *nodes* as context.

        Args:
            query: The user's original query string.
            nodes: Ranked and expanded context nodes.

        Returns:
            A :class:`GenerationResult` with answer, citations, and trace.
        """
        # Build citation-aware context (available for inspection)
        _ = self._build_context(nodes)

        response = await self.synthesizer.asynthesize(
            query,
            nodes=nodes,
        )

        return GenerationResult(
            answer=str(response),
            citations=self._extract_citations(nodes),
            retrieval_trace=self._build_trace(nodes),
        )

    def _build_context(self, nodes: list[NodeWithScore]) -> str:
        """Build a numbered, citation-aware context string.

        Args:
            nodes: Context nodes to format.

        Returns:
            Formatted context string with source annotations.
        """
        parts: list[str] = []
        for i, node in enumerate(nodes, 1):
            meta = node.metadata
            parts.append(
                f"[{i}] Source: {meta.get('section_path', 'unknown')} "
                f"(page {meta.get('page', '?')})\n{node.text}"
            )
        return "\n\n".join(parts)

    def _extract_citations(
        self,
        nodes: list[NodeWithScore],
    ) -> list[Citation]:
        """Extract structured citations from context nodes.

        Args:
            nodes: Context nodes used for generation.

        Returns:
            List of :class:`Citation` objects.
        """
        return [
            Citation(
                index=i,
                section_path=n.metadata.get("section_path", ""),
                page=n.metadata.get("page", 0),
                chunk_type=n.metadata.get("chunk_type", ""),
                score=n.score or 0.0,
            )
            for i, n in enumerate(nodes, 1)
        ]

    def _build_trace(
        self,
        nodes: list[NodeWithScore],
    ) -> list[dict[str, Any]]:
        """Build a retrieval trace for observability.

        Args:
            nodes: Context nodes used for generation.

        Returns:
            List of trace dicts with rank, score, section info.
        """
        return [
            {
                "rank": i,
                "node_id": n.node_id,
                "score": n.score,
                "section_path": n.metadata.get("section_path"),
                "depth": n.metadata.get("depth"),
                "page": n.metadata.get("page"),
                "chunk_type": n.metadata.get("chunk_type"),
            }
            for i, n in enumerate(nodes, 1)
        ]
