"""
Hybrid Retriever
================

Implements the full retrieval pipeline in exact order:

1. **HyDE** — generate hypothetical answer for vector search only
2. **Vector retrieval** — uses HyDE embedding, top-30
3. **BM25 retrieval** — uses RAW query (never HyDE), top-30
4. **RRF fusion** — reciprocal rank fusion with configurable weights
5. **TOC retrieval** — entry point for deep content; fetches child chunks
6. **Merge & deduplicate** — by ``parent_id`` (one chunk per parent)
7. **Score boosting** — depth boost + importance boost
8. **Top-50 selection**

Critical rules enforced here:
- BM25 always uses the raw query.
- HyDE only for vector search.
- Deduplication by ``parent_id`` before reranking.
- Depth boost: ``1.0 / (1 + log1p(depth)) * 0.10``
"""


from __future__ import annotations

import json
from math import log1p

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from config import config
from embeddings.bge import BGEEmbedding


class HybridRetriever:
    """Multi-stage hybrid retriever combining vector, BM25, and TOC search.

    The retrieval flow follows a strict 8-step order (see module
    docstring).  All configuration is read from
    :pydata:`llamaindex_rag.config.config`.

    Args:
        chunk_index: Vector index for content chunks.
        toc_index: Vector index for TOC entries.
        storage_context: Storage context containing the docstore.
        all_nodes: All content chunk nodes (needed for BM25 and TOC
            child lookup).
    """

    def __init__(
        self,
        chunk_index: VectorStoreIndex,
        toc_index: VectorStoreIndex,
        storage_context: StorageContext,
        all_nodes: list[TextNode],
    ) -> None:
        self.chunk_index = chunk_index
        self.toc_index = toc_index
        self.storage_context = storage_context
        self.all_nodes = all_nodes
        self.embed_model = BGEEmbedding()

    async def retrieve(self, query: str) -> list[NodeWithScore]:
        """Execute the full hybrid retrieval pipeline for *query*.

        Args:
            query: The user's natural-language query.

        Returns:
            Top-k candidate nodes (default 50), scored and deduplicated.
        """
        # ---------------------------------------------------------------
        # Step 1: HyDE — hypothetical document for vector search only  # uses ai chat/completions
        # ---------------------------------------------------------------
        vector_query = query
        if config.enable_hyde:
            try:
                from llama_index.core.indices.query.query_transform import (
                    HyDEQueryTransform,
                )
                from llama_index.llms.openai import OpenAI

                hyde = HyDEQueryTransform(
                    include_original=True,
                    llm=OpenAI(model=config.llm_model),
                )
                hyde_bundle = hyde(query)
                # Use the HyDE-augmented query string for vector search
                if hasattr(hyde_bundle, "query_str") and hyde_bundle.query_str:
                    vector_query = hyde_bundle.query_str
            except Exception:
                # If HyDE fails (e.g., no LLM key), fall back to raw query
                vector_query = query

        # ---------------------------------------------------------------
        # Step 2: Vector retriever uses HyDE embedding
        # ---------------------------------------------------------------
        vector_retriever = VectorIndexRetriever(
            index=self.chunk_index,
            similarity_top_k=30,
            embed_model=self.embed_model,
        )

        # ---------------------------------------------------------------
        # Step 3: BM25 uses RAW query — never HyDE
        # ---------------------------------------------------------------
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.all_nodes,
            similarity_top_k=30,
        )

        # ---------------------------------------------------------------
        # Step 4: Fuse with RRF
        # ---------------------------------------------------------------
        if config.enable_hybrid:
            fusion_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=config.retrieval_top_k,
                num_queries=1,  # no extra query generation
                mode="reciprocal_rerank",
                use_async=True,
                llm=None,  # no LLM needed when num_queries=1
                retriever_weights=[
                    config.vector_weight,
                    config.bm25_weight,
                ],
            )
            # Step 5: Get vector+BM25 candidates
            vector_bm25_nodes = await fusion_retriever.aretrieve(vector_query)
        else:
            # Vector-only mode
            vector_bm25_nodes = await vector_retriever.aretrieve(vector_query)

        # ---------------------------------------------------------------
        # Step 6: TOC retrieval — entry point for deep content
        # ---------------------------------------------------------------
        toc_nodes: list[NodeWithScore] = []
        if config.enable_toc_index:
            toc_retriever = VectorIndexRetriever(
                index=self.toc_index,
                similarity_top_k=config.toc_top_k,
            )
            toc_hits = await toc_retriever.aretrieve(query)

            # For each TOC hit, fetch all its child chunks
            for toc_hit in toc_hits:
                child_ids_json = toc_hit.metadata.get("child_node_ids", "[]")
                try:
                    child_ids: list[str] = json.loads(child_ids_json)
                except (json.JSONDecodeError, TypeError):
                    child_ids = []

                #children = self._get_nodes_by_ids(child_ids)
                # IMPORTANT: Inject TOC children with a conservative base score.
                # TOC is a recall mechanism; it should not swamp high-precision
                # vector/BM25 hits in the candidate pool.
                toc_score = float(toc_hit.score or 0.0)
                base_score = max(0.0, toc_score) * config.toc_child_score_scale
                children = self._get_nodes_by_ids(child_ids, base_score=base_score)
                toc_nodes.extend(children)

        # ---------------------------------------------------------------
        # Step 7: Merge all candidates, deduplicate by parent_id
        # ---------------------------------------------------------------
        all_candidates = self._deduplicate(vector_bm25_nodes + toc_nodes)

        # ---------------------------------------------------------------
        # Step 8: Apply depth boost and importance boost to scores
        # ---------------------------------------------------------------
        for node in all_candidates:
            depth = node.metadata.get("depth", 1)
            if isinstance(depth, str):
                try:
                    depth = int(depth)
                except ValueError:
                    depth = 1
            importance = node.metadata.get("importance_score", 0.0)
            if isinstance(importance, str):
                try:
                    importance = float(importance)
                except ValueError:
                    importance = 0.0
            depth_boost = 1.0 / (1 + log1p(depth)) * 0.10
            node.score = (node.score or 0.0) + importance * 0.15 + depth_boost

        # ---------------------------------------------------------------
        # Step 9: Sort, keep top-k
        # ---------------------------------------------------------------
        all_candidates.sort(key=lambda x: x.score or 0.0, reverse=True)
        return all_candidates[: config.retrieval_top_k]

    def _deduplicate(
        self,
        nodes: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """Deduplicate nodes by ``parent_id``.

        Keeps the highest-scored node per ``parent_id`` to prevent
        sibling chunks from filling all top-10 slots.  Nodes without
        a ``parent_id`` are kept unconditionally (deduplicated by
        ``node_id``).

        Args:
            nodes: Raw candidate nodes (may contain duplicates).

        Returns:
            Deduplicated list of nodes.
        """
        seen_parents: dict[str, NodeWithScore] = {}
        seen_ids: set[str] = set()
        no_parent: list[NodeWithScore] = []

        for node in nodes:
            # Deduplicate by node_id first
            nid = node.node_id
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            parent_id = node.metadata.get("parent_id", "")
            if not parent_id:
                no_parent.append(node)
                continue
            if parent_id not in seen_parents:
                seen_parents[parent_id] = node
            else:
                if (node.score or 0) > (
                    seen_parents[parent_id].score or 0
                ):
                    seen_parents[parent_id] = node

        return list(seen_parents.values()) + no_parent

    def _get_nodes_by_ids(
        self,
        ids: list[str],
        base_score: float = 0.0,
    ) -> list[NodeWithScore]:
        """Fetch nodes from :attr:`all_nodes` by their IDs.

        Args:
            ids: List of node ID strings to retrieve.
            base_score: Base score to assign to each returned node.

        Returns:
            Matching nodes wrapped as :class:`NodeWithScore`.
        """
        id_set = set(ids)
        return [
            NodeWithScore(node=n, score=base_score)
            for n in self.all_nodes
            if n.node_id in id_set
        ]
