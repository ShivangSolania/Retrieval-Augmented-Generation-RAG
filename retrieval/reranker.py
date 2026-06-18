"""
BGE Reranker
============

Cross-encoder reranker using ``BAAI/bge-reranker-base`` via LlamaIndex's
:class:`SentenceTransformerRerank` post-processor.

**Critical rules**:
- Reranking always uses the **original** query, never HyDE.
- Reranking always happens **before** parent expansion (AutoMerging).
"""

from __future__ import annotations

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle

from config import config


def BGEReranker() -> SentenceTransformerRerank:
    """Returns a native SentenceTransformerRerank configured for BGE."""
    return SentenceTransformerRerank(
        model=config.reranker_model,
        top_n=config.rerank_top_k,
    )
