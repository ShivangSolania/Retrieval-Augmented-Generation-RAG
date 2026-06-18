"""
RAG Pipeline Orchestrator
=========================

Wires all pipeline components together into a single
:class:`RAGPipeline` class that exposes ``ingest()`` and ``query()``
as the two primary operations.

Ingestion flow::

    Document → Docling parse → LlamaIndex nodes → Chunking router
    → TOC index builder → ChromaDB storage (two collections)

Query flow::

    Query → Hybrid retrieve (top-50) → Rerank (top-10)
    → Parent expansion → LLM generation → Result
"""

from __future__ import annotations

import time

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from chunking.router import ChunkingRouter
from config import config
from generation.generator import (
    Generator,
    IngestResult,
    QueryResult,
    RankedChunk,
    SearchResult,
)
from ingestion.parser import DoclingParser
from ingestion.toc import TOCIndexBuilder
from observability.logger import RAGLogger
from retrieval.expander import ContextExpander
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import BGEReranker
from storage.chroma import ChromaStore


class RAGPipeline:
    """Full RAG pipeline orchestrator.

    Combines parsing, chunking, embedding, storage, retrieval, reranking,
    parent expansion, and generation into two high-level operations:

    * :meth:`ingest` — parse and store a document
    * :meth:`search` — hybrid retrieve + rerank (top-k, no LLM)
    * :meth:`query` — retrieve context and generate an answer

    Usage::

        pipeline = RAGPipeline()
        await pipeline.ingest("report.pdf", "doc-001")
        hits = await pipeline.search("What are the key findings?")
        result = await pipeline.query("What are the key findings?")
        print(result.answer)
    """

    def __init__(self) -> None:
        self.store = ChromaStore()
        self.parser = DoclingParser()
        self.toc_builder = TOCIndexBuilder()
        self.chunker = ChunkingRouter()
        self.reranker = BGEReranker()
        self.generator = Generator()
        self.logger = RAGLogger()

        self.chunk_index: VectorStoreIndex | None = None
        self.toc_index: VectorStoreIndex | None = None
        self.all_nodes: list[TextNode] = []

    def _ensure_loaded_from_store(self) -> bool:
        """Load persisted indices/nodes from Chroma when memory is empty.

        Returns:
            ``True`` when indices are available after this call.
        """
        if self.chunk_index is not None and self.toc_index is not None:
            return True

        if self.store.chunks_collection.count() == 0:
            return False

        self.chunk_index = self.store.load_chunk_index()
        self.toc_index = self.store.load_toc_index()
        self.all_nodes = self.store.load_all_chunk_nodes()
        return True

    async def ingest(self, file_path: str, doc_id: str) -> IngestResult:
        """Parse, chunk, and store a document.

        Steps:
        1. Parse with Docling
        2. Convert to LlamaIndex nodes
        3. Chunk via router (Hierarchical / Markdown / Semantic)
        4. Build TOC nodes
        5. Store in ChromaDB (chunks + toc_index collections)
        6. BM25 index is rebuilt from ``self.all_nodes``

        Args:
            file_path: Path to the document file.
            doc_id: Unique document identifier.

        Returns:
            An :class:`IngestResult` with ingestion metrics.
        """
        t0 = time.time()

        # Step 1: Parse with Docling
        parsed = self.parser.parse(file_path)

        # Step 2: Convert to LlamaIndex nodes
        base_nodes = self.parser.to_llama_nodes(parsed, doc_id)

        # Step 3: Chunk via router
        chunked_nodes = self.chunker.route(file_path, base_nodes)

        # Step 4: Build TOC nodes
        toc_nodes = self.toc_builder.build(parsed.toc, chunked_nodes)

        # Step 5: Store in ChromaDB
        self.chunk_index = self.store.build_chunk_index(chunked_nodes)
        self.toc_index = self.store.build_toc_index(toc_nodes)

        # Step 6: Update in-memory node list (BM25 rebuilt on every ingest)
        self.all_nodes = chunked_nodes

        latency_ms = int((time.time() - t0) * 1000)

        latency_sec = latency_ms / 1000

        self.logger.log_ingestion(
            doc_id=doc_id,
            chunk_count=len(chunked_nodes),
            toc_count=len(toc_nodes),
            latency_ms=latency_ms,
            latency_sec=latency_sec,
        )

        return IngestResult(
            doc_id=doc_id,
            chunk_count=len(chunked_nodes),
            toc_entries=len(toc_nodes),
            ingestion_time_ms=latency_ms,
            ingestion_time_sec=latency_sec,
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> SearchResult:
        """Hybrid retrieve and rerank — no parent expansion or LLM.

        Loads indices from ChromaDB when this process has not ingested
        in-memory yet (same as :meth:`query`).

        Args:
            query: Natural-language search query.
            top_k: Number of chunks to return (default ``config.rerank_top_k``).

        Returns:
            Top-k ranked chunks and retrieval trace.

        Raises:
            RuntimeError: If no documents have been ingested yet.
        """
        if not self._ensure_loaded_from_store():
            raise RuntimeError(
                "No documents ingested. Call ingest() before search()."
            )

        k = top_k if top_k is not None else config.rerank_top_k
        t0 = time.time()

        retriever = HybridRetriever(
            chunk_index=self.chunk_index,
            toc_index=self.toc_index,
            storage_context=self.store.storage_context,
            all_nodes=self.all_nodes,
        )

        candidates = await retriever.retrieve(query)

        if config.enable_reranking:
            from llama_index.core.schema import QueryBundle
            reranked = self.reranker.postprocess_nodes(candidates, QueryBundle(query_str=query))
        else:
            reranked = candidates[:k]

        top = reranked[:k]
        trace = self.generator._build_trace(top)

        latency_ms = int((time.time() - t0) * 1000)
        latency_sec = latency_ms / 1000

        self.logger.log_retrieval(
            query=query,
            candidates=candidates,
            reranked=top,
            latency_ms=latency_ms,
            latency_sec=latency_sec,
        )

        return SearchResult(
            chunks=[
                RankedChunk(
                    text=n.text or "",
                    score=n.score or 0.0,
                    metadata=dict(n.metadata or {}),
                    node_id=n.node_id,
                )
                for n in top
            ],
            trace=trace,
            latency_ms=latency_ms,
            latency_sec=latency_sec,
        )

    async def query(self, query: str) -> QueryResult:
        """Retrieve context and generate an answer for *query*.

        Steps:
        1. Hybrid retrieve (top-50 candidates)
        2. Cross-encoder rerank (top-10)
        3. Parent expansion via AutoMergingRetriever
        4. LLM generation with citations

        Args:
            query: The user's natural-language question.

        Returns:
            A :class:`QueryResult` with answer, citations, trace,
            and latency.

        Raises:
            RuntimeError: If no documents have been ingested yet.
        """
        if not self._ensure_loaded_from_store():
            raise RuntimeError(
                "No documents ingested. Call ingest() before query()."
            )

        t0 = time.time()

        # Build retriever for this query
        retriever = HybridRetriever(
            chunk_index=self.chunk_index,
            toc_index=self.toc_index,
            storage_context=self.store.storage_context,
            all_nodes=self.all_nodes,
        )

        # Step 1: Retrieve top-50
        candidates = await retriever.retrieve(query)

        # Step 2: Rerank to top-10
        if config.enable_reranking:
            from llama_index.core.schema import QueryBundle
            reranked = self.reranker.postprocess_nodes(candidates, QueryBundle(query_str=query))
        else:
            reranked = candidates[: config.rerank_top_k]

        # Step 3: Expand parent context
        if config.enable_parent_expansion and self.store.storage_context:
            expander = ContextExpander(
                chunk_index=self.chunk_index,
                storage_context=self.store.storage_context,
            )
            expanded = expander.expand(query, reranked)
        else:
            expanded = reranked

        # Step 4: Generate answer
        result = await self.generator.generate(query, expanded)

        latency_ms = int((time.time() - t0) * 1000)

        latency_sec = latency_ms / 1000

        self.logger.log_retrieval(
            query=query,
            candidates=candidates,
            reranked=reranked,
            latency_ms=latency_ms,
            latency_sec=latency_sec,
        )

        self.logger.log_generation(
            query=query,
            chunks_used=len(expanded),
            latency_ms=latency_ms,
            latency_sec=latency_sec,
        )

        return QueryResult(
            answer=result.answer,
            citations=result.citations,
            trace=result.retrieval_trace,
            latency_ms=latency_ms,
            latency_sec=latency_sec,
        )
