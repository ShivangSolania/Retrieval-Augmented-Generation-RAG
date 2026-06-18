"""
BGE Embedding Wrapper
=====================

Extends LlamaIndex's :class:`HuggingFaceEmbedding` to automatically apply
the BGE query prefix (``"Represent this sentence for retrieval: …"``),
which is required for optimal retrieval accuracy with BGE models.
"""

from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import config

def BGEEmbedding() -> HuggingFaceEmbedding:
    """Returns a native HuggingFaceEmbedding configured for BGE with query prefix."""
    return HuggingFaceEmbedding(
        model_name=config.embedding_model,
        embed_batch_size=32,
        query_instruction="Represent this sentence for retrieval: "
    )