"""
FastAPI Application
===================

Exposes the RAG pipeline as a REST API with five endpoints:

* ``POST /ingest`` — parse and store a document
* ``POST /search`` — retrieve and rerank chunks (no generation)
* ``POST /query`` — full RAG pipeline (retrieve + generate)
* ``GET  /health`` — system status and collection counts
* ``GET  /diagnostics/{doc_id}`` — per-document chunk diagnostics
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation.generator import IngestResult, QueryResult, SearchResult
from pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# App & pipeline
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LlamaIndex RAG API",
    description=(
        "Production RAG pipeline with Docling parsing, hybrid retrieval, "
        "BGE reranking, and GPT-4o generation."
    ),
    version="1.0.0",
)

pipeline = RAGPipeline()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for the ``/ingest`` endpoint.

    Attributes:
        file_path: Absolute path to the document on the server.
        doc_id: Unique document identifier.
    """

    file_path: str
    doc_id: str


class SearchRequest(BaseModel):
    """Request body for the ``/search`` endpoint.

    Attributes:
        query: Natural-language search query.
        top_k: Number of results to return after reranking.
    """

    query: str
    top_k: int = 10


class QueryRequest(BaseModel):
    """Request body for the ``/query`` endpoint.

    Attributes:
        query: Natural-language question.
    """

    query: str


class HealthResponse(BaseModel):
    """Response from the ``/health`` endpoint.

    Attributes:
        status: Always ``"ok"`` if the server is running.
        chunks_count: Number of entries in the chunks collection.
        toc_count: Number of entries in the TOC index collection.
        bm25_index_size: Number of nodes in the in-memory BM25 index.
    """

    status: str = "ok"
    chunks_count: int = 0
    toc_count: int = 0
    bm25_index_size: int = 0


class DiagnosticsResponse(BaseModel):
    """Response from the ``/diagnostics/{doc_id}`` endpoint.

    Attributes:
        chunk_count: Total chunks for the document.
        depth_distribution: Count of chunks at each depth level.
        avg_importance: Mean importance score across all chunks.
    """

    chunk_count: int = 0
    depth_distribution: dict[str, int] = Field(default_factory=dict)
    avg_importance: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResult)
async def ingest(request: IngestRequest) -> IngestResult:
    """Ingest a document: parse, chunk, embed, and store.

    Args:
        request: Ingest request with file path and document ID.

    Returns:
        Ingestion metrics (chunk count, TOC entries, time).
    """
    return await pipeline.ingest(request.file_path, request.doc_id)


@app.post("/search", response_model=SearchResult)
async def search(request: SearchRequest) -> SearchResult:
    """Search for relevant chunks without generating an answer.

    Args:
        request: Search request with query and top_k.

    Returns:
        Ranked chunks and retrieval trace.

    Raises:
        HTTPException: If no documents have been ingested.
    """
    try:
        return await pipeline.search(request.query, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResult)
async def query(request: QueryRequest) -> QueryResult:
    """Full RAG query: retrieve context and generate an answer.

    Args:
        request: Query request with the user's question.

    Returns:
        Answer, citations, retrieval trace, and latency.
    """
    return await pipeline.query(request.query)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check and system status.

    Returns:
        Collection counts and BM25 index size.
    """
    pipeline._ensure_loaded_from_store()
    return HealthResponse(
        status="ok",
        chunks_count=pipeline.store.chunks_collection.count(),
        toc_count=pipeline.store.toc_collection.count(),
        bm25_index_size=len(pipeline.all_nodes),
    )


@app.get("/diagnostics/{doc_id}", response_model=DiagnosticsResponse)
async def diagnostics(doc_id: str) -> DiagnosticsResponse:
    """Per-document chunk diagnostics.

    Args:
        doc_id: The document ID to inspect.

    Returns:
        Chunk count, depth distribution, and average importance score.
    """
    nodes = [
        n
        for n in pipeline.all_nodes
        if n.metadata.get("doc_id") == doc_id
    ]
    if not nodes:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for doc_id='{doc_id}'.",
        )

    depths = [n.metadata.get("depth", 0) for n in nodes]
    depth_dist = dict(Counter(str(d) for d in depths))
    avg_importance = sum(
        n.metadata.get("importance_score", 0.0) for n in nodes
    ) / max(len(nodes), 1)

    return DiagnosticsResponse(
        chunk_count=len(nodes),
        depth_distribution=depth_dist,
        avg_importance=round(avg_importance, 4),
    )
