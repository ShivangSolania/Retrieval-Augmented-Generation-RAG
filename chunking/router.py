"""
Chunking Router
===============

Routes documents to the appropriate LlamaIndex node parser based on
file type and content characteristics:

* ``.md`` files → :class:`MarkdownNodeParser`
* Docs with clear headings (>20 % heading blocks) → :class:`HierarchicalNodeParser`
* Default → :class:`HierarchicalNodeParser` (three-level: 1024 / 512 / 256)
* Dense prose fallback (<5 % headings, not markdown) → :class:`SemanticSplitterNodeParser`

After parsing, every child node inherits its parent's section metadata
so that deep chunks are never orphaned from their context.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import TextNode

from config import config
from embeddings.bge import BGEEmbedding

# Metadata keys that must propagate from parent to child nodes
_PROPAGATED_KEYS = (
    "section_path",
    "section_title",
    "depth",
    "importance_score",
    "toc_section_id",
    "doc_id",
    "source_file",
    "chunk_type",
)


class ChunkingRouter:
    """Selects and applies the best chunking strategy per document.

    The three-level :class:`HierarchicalNodeParser` hierarchy
    (``parent_chunk_size`` / ``chunk_size`` / 256) is what
    :class:`AutoMergingRetriever` relies on at retrieval time.

    Usage::

        router = ChunkingRouter()
        chunked = router.route("report.pdf", base_nodes)
    """

    def __init__(self) -> None:
        self._embed_model = BGEEmbedding()

    def route(
        self,
        file_path: str,
        nodes: list[TextNode],
    ) -> list[TextNode]:
        """Choose a parser and chunk *nodes* from *file_path*.

        Args:
            file_path: Path to the source file (used for extension routing).
            nodes: Base :class:`TextNode` objects from the Docling parser.

        Returns:
            Chunked nodes with full metadata propagated from parents.
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".md":
            chunked = self._markdown_chunk(nodes)
        elif self._is_dense_prose(nodes, ext):
            chunked = self._semantic_chunk(nodes)
        else:
            # Default: hierarchical (covers heading-rich docs too)
            chunked = self._hierarchical_chunk(nodes)

        # Propagate section metadata from original nodes to children
        self._propagate_metadata(nodes, chunked)

        return chunked

    # ------------------------------------------------------------------
    # Chunking strategies
    # ------------------------------------------------------------------

    def _hierarchical_chunk(self, nodes: list[TextNode]) -> list[TextNode]:
        """Apply three-level hierarchical chunking.

        Chunk sizes: ``[parent_chunk_size, chunk_size, 256]`` — the
        structure :class:`AutoMergingRetriever` needs.

        Args:
            nodes: Base nodes to chunk.

        Returns:
            Hierarchically chunked nodes.
        """
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[
                config.parent_chunk_size,  # 1024 — parent nodes
                config.chunk_size,         # 512  — child nodes
                256,                       # leaf nodes
            ],
        )
        return parser.get_nodes_from_documents(nodes)

    def _markdown_chunk(self, nodes: list[TextNode]) -> list[TextNode]:
        """Apply Markdown-aware chunking.

        Args:
            nodes: Base nodes to chunk.

        Returns:
            Markdown-split nodes.
        """
        parser = MarkdownNodeParser()
        return parser.get_nodes_from_documents(nodes)

    def _semantic_chunk(self, nodes: list[TextNode]) -> list[TextNode]:
        """Apply semantic splitting for dense prose.

        Uses the BGE embedding model with a breakpoint percentile
        threshold derived from ``config.semantic_threshold``.

        Args:
            nodes: Base nodes to chunk.

        Returns:
            Semantically split nodes.
        """
        parser = SemanticSplitterNodeParser(
            embed_model=self._embed_model,
            breakpoint_percentile_threshold=int(
                config.semantic_threshold * 100
            ),
        )
        return parser.get_nodes_from_documents(nodes)

    # ------------------------------------------------------------------
    # Content analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_clear_headings(nodes: list[TextNode]) -> bool:
        """Return ``True`` if >20 % of *nodes* are headings.

        Args:
            nodes: Nodes to analyse.

        Returns:
            Whether the document has clear heading structure.
        """
        if not nodes:
            return False
        heading_count = sum(
            1 for n in nodes if n.metadata.get("chunk_type") == "heading"
        )
        return heading_count / len(nodes) > 0.20

    @staticmethod
    def _is_dense_prose(nodes: list[TextNode], ext: str) -> bool:
        """Return ``True`` if content is dense prose without structure.

        Criteria: <5 % headings AND not a Markdown file.

        Args:
            nodes: Nodes to analyse.
            ext: File extension (lowercase, with dot).

        Returns:
            Whether the document is dense, unstructured prose.
        """
        if ext == ".md":
            return False
        if not nodes:
            return False
        heading_count = sum(
            1 for n in nodes if n.metadata.get("chunk_type") == "heading"
        )
        return heading_count / len(nodes) < 0.05

    # ------------------------------------------------------------------
    # Metadata propagation
    # ------------------------------------------------------------------

    @staticmethod
    def _propagate_metadata(
        original_nodes: list[TextNode],
        chunked_nodes: list[TextNode],
    ) -> None:
        """Copy section metadata from *original_nodes* into *chunked_nodes*.

        After parsing, child nodes may lose their section context.
        This method restores it by matching each child back to its
        source node (via text overlap) and copying the propagated keys.

        Args:
            original_nodes: The pre-chunking nodes with full metadata.
            chunked_nodes: The post-chunking nodes to enrich.
        """
        if not original_nodes:
            return

        for child in chunked_nodes:
            # If child already has section_path set, skip
            if child.metadata.get("section_path"):
                continue

            # Find best matching original node by text overlap
            best_match: TextNode | None = None
            best_overlap = 0
            child_text = child.text[:200]  # Use prefix for matching

            for orig in original_nodes:
                # Check if child text is a substring of original
                if child_text in orig.text:
                    overlap = len(child_text)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = orig

            if best_match is None and original_nodes:
                # Fallback: use first original node
                best_match = original_nodes[0]

            if best_match is not None:
                for key in _PROPAGATED_KEYS:
                    if key not in child.metadata or not child.metadata[key]:
                        value = best_match.metadata.get(key)
                        if value is not None:
                            child.metadata[key] = value
