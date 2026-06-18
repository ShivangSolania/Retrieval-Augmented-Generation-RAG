"""
Context Expander (Parent Expansion)
====================================

Uses LlamaIndex's :class:`AutoMergingRetriever` to replace clusters
of child chunks with their full parent node when more than 50 % of a
parent's children appear in the reranked results.

This is always the **last** retrieval step — it runs **after** reranking.
"""

from __future__ import annotations

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import (
    AutoMergingRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle

from config import config


class ContextExpander:
    """Expands reranked results by merging child chunks into parent nodes.

    When >50 % of a parent node's children are present in the reranked
    results, :class:`AutoMergingRetriever` replaces them with the full
    parent node to provide broader context to the LLM.

    Args:
        chunk_index: The vector index for content chunks.
        storage_context: Storage context that includes the
            :class:`SimpleDocumentStore` populated with all nodes.

    Usage::

        expander = ContextExpander(chunk_index, storage_context)
        expanded = expander.expand("user query", reranked_nodes)
    """

    def __init__(
        self,
        chunk_index: VectorStoreIndex,
        storage_context: StorageContext,
    ) -> None:
        base_retriever = VectorIndexRetriever(
            index=chunk_index,
            similarity_top_k=config.rerank_top_k,
        )
        self.auto_merging = AutoMergingRetriever(
            vector_retriever=base_retriever,
            storage_context=storage_context,
            simple_ratio_thresh=0.5,
        )

    def expand(
        self,
        query: str,
        reranked_nodes: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """Expand *reranked_nodes* by merging children into parents.

        Feeds the reranked nodes through :class:`AutoMergingRetriever`
        which transparently replaces child clusters with the full
        parent node where appropriate.

        Args:
            query: The user's original query string.
            reranked_nodes: Nodes after cross-encoder reranking.

        Returns:
            Expanded nodes with parent context where applicable.
        """
        if not reranked_nodes:
            return []

        docstore = self.auto_merging._storage_context.docstore
        expanded_nodes = []
        seen_parents = set()
        
        for n in reranked_nodes:
            parent_id = n.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                try:
                    parent_node = docstore.get_document(parent_id)
                    if parent_node:
                        expanded_nodes.append(NodeWithScore(node=parent_node, score=n.score))
                        seen_parents.add(parent_id)
                        continue
                except ValueError:
                    pass
            elif parent_id in seen_parents:
                continue
            expanded_nodes.append(n)

        return expanded_nodes
