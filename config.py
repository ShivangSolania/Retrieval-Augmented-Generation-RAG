"""
Centralized Configuration
=========================

Pydantic-settings based configuration for the entire RAG pipeline.
Reads from environment variables and ``.env`` file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve .env path relative to this file (the package directory),
# so it works regardless of the working directory.
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class RAGConfig(BaseSettings):
    """Root configuration for the LlamaIndex RAG pipeline.

    All values can be overridden via environment variables or a ``.env`` file.
    For example, setting ``CHUNK_SIZE=256`` in the environment overrides the default.

    Attributes:
        chunk_size: Target chunk size in tokens for child nodes.
        chunk_overlap: Overlap between adjacent chunks.
        parent_chunk_size: Parent chunk size for AutoMerging hierarchy.
        semantic_threshold: Cosine-similarity breakpoint for semantic chunking.
        retrieval_top_k: Number of candidates to retrieve before reranking.
        rerank_top_k: Number of results after cross-encoder reranking.
        bm25_weight: Weight for BM25 scores in RRF fusion.
        vector_weight: Weight for vector similarity scores in RRF fusion.
        rrf_k: Reciprocal Rank Fusion constant.
        toc_top_k: Number of TOC entries to retrieve per query.
        toc_child_score_scale: Scale factor for injected TOC child chunk scores.
        embedding_model: HuggingFace model name for dense embeddings.
        reranker_model: HuggingFace model name for cross-encoder reranking.
        llm_model: OpenAI model name for generation.
        transcription_model: OpenAI model for audio transcription (diarized).
        enable_hyde: Whether to use HyDE (Hypothetical Document Embeddings).
        enable_hybrid: Whether to use hybrid (vector + BM25) retrieval.
        enable_reranking: Whether to apply cross-encoder reranking.
        enable_parent_expansion: Whether to use AutoMergingRetriever.
        enable_toc_index: Whether to use the separate TOC index layer.
    """

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 1024
    semantic_threshold: float = 0.82

    # --- Retrieval ---
    retrieval_top_k: int = 50
    rerank_top_k: int = 10
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    rrf_k: int = 60
    toc_top_k: int = 5
    toc_child_score_scale: float = 0.08

    # --- Models ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_model: str = "gpt-4o-mini"
    transcription_model: str = "whisper-1"

    # --- Feature Flags ---
    enable_hyde: bool = True
    enable_hybrid: bool = True
    enable_reranking: bool = True
    enable_parent_expansion: bool = True
    enable_toc_index: bool = True

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"
        extra = "ignore"


config = RAGConfig()
