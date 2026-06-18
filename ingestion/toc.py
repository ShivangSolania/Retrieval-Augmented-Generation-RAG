"""
TOC Index Builder
=================

Creates dedicated :class:`TextNode` objects for every Table-of-Contents
entry in a document.  These nodes are stored in a **separate** ChromaDB
collection (``"toc_index"``) and queried at retrieval time to inject
child chunks from deep sections that vector search alone would miss.

This is the critical fix for deep document retrieval — without it,
content past page 30 in long documents routinely gets missed.
"""

from __future__ import annotations

import json

from llama_index.core.schema import TextNode

from ingestion.parser import TOCEntry


class TOCIndexBuilder:
    """Builds a searchable TOC index layer from parsed TOC entries.

    For each :class:`TOCEntry` a dedicated :class:`TextNode` is created
    whose ``text`` encodes the section path, title, and a short summary
    (first 200 characters of section content).  The metadata carries a
    JSON-encoded list of child node IDs so the retriever can expand a
    TOC hit into the actual content chunks.

    Usage::

        builder = TOCIndexBuilder()
        toc_nodes = builder.build(parsed.toc, chunked_nodes)
    """

    def build(
        self,
        toc_entries: list[TOCEntry],
        all_nodes: list[TextNode],
    ) -> list[TextNode]:
        """Build TOC index nodes from TOC entries and content nodes.

        Args:
            toc_entries: Table-of-contents entries from the parser.
            all_nodes: All chunked content nodes (already processed by
                the chunking router).

        Returns:
            List of :class:`TextNode` objects — one per TOC entry — ready
            for insertion into the ``"toc_index"`` ChromaDB collection.
        """
        toc_nodes: list[TextNode] = []

        for entry in toc_entries:
            # Gather child node IDs and section text
            child_ids = self._get_children(entry, all_nodes)
            summary = self._build_summary(entry, all_nodes)

            # CRITICAL: text always includes section context
            node_text = f"{entry.section_path}\n{entry.title}\n{summary}"

            toc_node = TextNode(
                text=node_text,
                metadata={
                    "toc_id": entry.section_id,
                    "section_title": entry.title,
                    "section_path": entry.section_path,
                    "page": entry.page,
                    "depth": entry.level,
                    "child_node_ids": json.dumps(child_ids),
                    "is_toc_entry": True,
                },
            )
            toc_nodes.append(toc_node)

        return toc_nodes

    def _get_children(
        self,
        entry: TOCEntry,
        all_nodes: list[TextNode],
    ) -> list[str]:
        """Return node IDs of all content chunks belonging to *entry*.

        A node belongs to a TOC entry when its ``toc_section_id``
        metadata matches the entry's ``section_id``.

        Args:
            entry: The TOC entry to find children for.
            all_nodes: All content chunk nodes.

        Returns:
            List of node ID strings.
        """
        return [
            n.node_id
            for n in all_nodes
            if n.metadata.get("toc_section_id") == entry.section_id
        ]

    def _build_summary(
        self,
        entry: TOCEntry,
        all_nodes: list[TextNode],
    ) -> str:
        """Build a short summary of the section content (first 200 chars).

        Concatenates the text of all child nodes for *entry* and
        returns the first 200 characters.

        Args:
            entry: The TOC entry to summarise.
            all_nodes: All content chunk nodes.

        Returns:
            Summary string (max 200 characters).
        """
        section_texts: list[str] = []
        for node in all_nodes:
            if node.metadata.get("toc_section_id") == entry.section_id:
                section_texts.append(node.text)
        combined = " ".join(section_texts)
        return combined[:200] if combined else entry.title
