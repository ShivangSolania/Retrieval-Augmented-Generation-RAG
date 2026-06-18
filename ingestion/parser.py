"""
Docling Document Parser
=======================

Parses PDF, DOCX, PPTX, Markdown, and TXT files using Docling's
:class:`DocumentConverter`, then converts the structured output into
LlamaIndex :class:`TextNode` objects with rich metadata.

**Critical invariant**: every node's ``text`` field is always
``f"{section_path}\\n{section_title}\\n{block.text}"`` — never raw
block text alone.  This is what makes deep chunks retrievable.
"""

from __future__ import annotations

import re
import uuid
from math import log1p
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter
from llama_index.core.schema import TextNode
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class ContentBlock(BaseModel):
    """A single content block extracted from a parsed document.

    Attributes:
        type: Block type — one of heading, paragraph, table, list, code.
        text: The textual content of the block.
        level: Heading depth (1-6). Defaults to 0 for non-heading blocks.
        page: 1-indexed page number where this block appears.
    """

    type: str = "paragraph"
    text: str = ""
    level: int = 0
    page: int = 1


class TOCEntry(BaseModel):
    """A single Table-of-Contents entry.

    Attributes:
        title: Section heading text.
        level: Heading depth (1-6).
        page: Page number where the section begins.
        section_id: Unique slug derived from title + page.
        section_path: Hierarchical path like "Ch 1 > Sec 2 > Sub 3".
    """

    title: str
    level: int
    page: int
    section_id: str = ""
    section_path: str = ""


class ParsedDocument(BaseModel):
    """Result of parsing a single document with Docling.

    Attributes:
        title: Document title (extracted or derived from filename).
        blocks: Ordered list of content blocks.
        toc: Table-of-contents entries extracted from headings.
        source_file: Absolute path of the original file.
    """

    title: str = ""
    blocks: list[ContentBlock] = Field(default_factory=list)
    toc: list[TOCEntry] = Field(default_factory=list)
    source_file: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert *text* into a URL-safe slug.

    Args:
        text: Arbitrary unicode string.

    Returns:
        Lowercased, hyphen-separated ASCII slug.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80]


def compute_importance(block: ContentBlock) -> float:
    """Compute an importance score for *block* based on its type and depth.

    Scoring rules (additive, capped at 1.0):
    - heading: +0.30
    - table:   +0.25
    - code:    +0.20
    - depth boost: ``1 / (1 + log1p(level)) * 0.25``

    Args:
        block: The content block to score.

    Returns:
        Float in [0.0, 1.0].
    """
    score = 0.0
    if block.type == "heading":
        score += 0.30
    if block.type == "table":
        score += 0.25
    if block.type == "code":
        score += 0.20
    depth_boost = 1.0 / (1 + log1p(block.level)) * 0.25
    score += depth_boost
    return min(score, 1.0)


def nearest_heading_above(blocks: list[ContentBlock], current_index: int) -> str:
    """Walk backwards from *current_index* to find the nearest heading.

    Args:
        blocks: Full ordered list of content blocks.
        current_index: Index of the block whose heading we want.

    Returns:
        Text of the nearest heading above, or ``"Untitled Section"``.
    """
    for i in range(current_index, -1, -1):
        if blocks[i].type == "heading":
            return blocks[i].text
    return "Untitled Section"


def build_section_path(blocks: list[ContentBlock], current_index: int) -> str:
    """Build a hierarchical section path for the block at *current_index*.

    Walks backwards collecting headings at each depth level to produce a
    path like ``"Chapter 1 > Section 2 > Subsection 3"``.

    Args:
        blocks: Full ordered list of content blocks.
        current_index: Index of the block whose path we want.

    Returns:
        ``" > "``-joined section path string.
    """
    path_parts: dict[int, str] = {}
    for i in range(current_index, -1, -1):
        blk = blocks[i]
        if blk.type == "heading" and blk.level not in path_parts:
            path_parts[blk.level] = blk.text
    if not path_parts:
        return "Document Root"
    sorted_levels = sorted(path_parts.keys())
    return " > ".join(path_parts[lvl] for lvl in sorted_levels)


def nearest_toc_entry(
    toc_entries: list[TOCEntry],
    block: ContentBlock,
) -> TOCEntry:
    """Find the TOC entry closest to *block* by page and level.

    Args:
        toc_entries: All TOC entries from the document.
        block: The content block to match.

    Returns:
        Best-matching :class:`TOCEntry`, or a fallback entry.
    """
    if not toc_entries:
        return TOCEntry(title="Unknown", level=0, page=0, section_id="unknown")
    best: Optional[TOCEntry] = None
    best_distance = float("inf")
    for entry in toc_entries:
        if entry.page <= block.page:
            distance = block.page - entry.page
            if distance < best_distance:
                best_distance = distance
                best = entry
    return best or toc_entries[0]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


class DoclingParser:
    """Parses documents using Docling and converts them to LlamaIndex nodes.

    Supported formats: PDF, DOCX, PPTX, Markdown (.md), plain text (.txt).

    Usage::

        parser = DoclingParser()
        parsed = parser.parse("report.pdf")
        nodes = parser.to_llama_nodes(parsed, doc_id="rpt-001")
    """

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def parse(self, file_path: str) -> ParsedDocument:
        """Parse a document file and extract structured content.

        Args:
            file_path: Absolute or relative path to the document.

        Returns:
            A :class:`ParsedDocument` with blocks, TOC entries, and title.
        """
        result = self._converter.convert(file_path)
        doc = result.document

        # --- Extract title ---
        title = ""
        if hasattr(doc, "title") and doc.title:
            title = str(doc.title)
        if not title:
            title = Path(file_path).stem

        # --- Extract blocks ---
        blocks: list[ContentBlock] = []
        self._extract_blocks(doc, blocks)

        # --- If no blocks extracted, fall back to full text ---
        if not blocks:
            full_text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
            if full_text:
                for i, paragraph in enumerate(full_text.split("\n\n")):
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    block_type = "heading" if paragraph.startswith("#") else "paragraph"
                    level = 0
                    if block_type == "heading":
                        level = min(paragraph.count("#", 0, 7), 6)
                        paragraph = paragraph.lstrip("#").strip()
                    blocks.append(ContentBlock(
                        type=block_type,
                        text=paragraph,
                        level=level,
                        page=1,
                    ))

        # --- Build TOC from headings ---
        heading_stack: dict[int, str] = {}
        toc_entries: list[TOCEntry] = []
        for block in blocks:
            if block.type == "heading":
                heading_stack[block.level] = block.text
                # Remove deeper levels
                keys_to_remove = [k for k in heading_stack if k > block.level]
                for k in keys_to_remove:
                    del heading_stack[k]
                sorted_levels = sorted(heading_stack.keys())
                section_path = " > ".join(heading_stack[lvl] for lvl in sorted_levels)
                section_id = f"{_slugify(block.text)}-p{block.page}"
                toc_entries.append(TOCEntry(
                    title=block.text,
                    level=block.level,
                    page=block.page,
                    section_id=section_id,
                    section_path=section_path,
                ))

        return ParsedDocument(
            title=title,
            blocks=blocks,
            toc=toc_entries,
            source_file=str(Path(file_path).resolve()),
        )

    def _extract_blocks(
        self,
        doc: object,
        blocks: list[ContentBlock],
    ) -> None:
        """Extract content blocks from a Docling document object.

        Handles various Docling document structures defensively.

        Args:
            doc: The Docling ``DoclingDocument`` object.
            blocks: Output list to append blocks to.
        """
        # Try iterating over document items (Docling v2.x API)
        items = None
        if hasattr(doc, "iterate_items"):
            try:
                items = list(doc.iterate_items())
            except Exception:
                items = None

        if items:
            for item_tuple in items:
                # iterate_items yields (item, level) or just item
                if isinstance(item_tuple, tuple):
                    item = item_tuple[0]
                else:
                    item = item_tuple

                block = self._item_to_block(item, doc)
                if block and block.text.strip():
                    blocks.append(block)
            return

        # Fallback: try body children
        if hasattr(doc, "body") and hasattr(doc.body, "children"):
            for child in doc.body.children:
                block = self._item_to_block(child, doc)
                if block and block.text.strip():
                    blocks.append(block)
            return

        # Fallback: try texts property
        if hasattr(doc, "texts"):
            for text_item in doc.texts:
                block = self._item_to_block(text_item, doc)
                if block and block.text.strip():
                    blocks.append(block)

    def _item_to_block(
        self,
        item: object,
        doc: object | None = None,
    ) -> Optional[ContentBlock]:
        """Convert a single Docling item to a :class:`ContentBlock`.

        Args:
            item: A Docling content item (heading, paragraph, table, etc.).

        Returns:
            A :class:`ContentBlock` or ``None`` if the item cannot be converted.
        """
        text = ""
        block_type = "paragraph"
        level = 0
        page = 1

        # Extract text — try .text first, but fall through to
        # export_to_markdown() when .text is empty (common for tables).
        if hasattr(item, "text") and str(item.text).strip():
            text = str(item.text)
        elif hasattr(item, "export_to_markdown"):
            try:
                # Some Docling items (e.g., PictureItem) require the parent
                # document as an argument for markdown export.
                text = str(item.export_to_markdown(doc))
            except TypeError:
                # Other item types expose a zero-argument signature.
                text = str(item.export_to_markdown())
        elif hasattr(item, "content"):
            text = str(item.content)

        if not text:
            return None

        # Determine type
        item_type = ""
        if hasattr(item, "label"):
            item_type = str(item.label).lower()
        elif hasattr(item, "obj_type"):
            item_type = str(item.obj_type).lower()
        elif hasattr(item, "content_type"):
            item_type = str(item.content_type).lower()

        if "heading" in item_type or "title" in item_type or "section" in item_type:
            block_type = "heading"
        elif "table" in item_type:
            block_type = "table"
        elif "list" in item_type:
            block_type = "list"
        elif "code" in item_type:
            block_type = "code"
        else:
            block_type = "paragraph"

        # Extract heading level
        if block_type == "heading":
            if hasattr(item, "level"):
                level = int(item.level) if item.level else 1
            else:
                # Try to detect from label like "section_header_level_2"
                level_match = re.search(r"(\d+)", item_type)
                level = int(level_match.group(1)) if level_match else 1
            level = max(1, min(level, 6))

        # Extract page number
        if hasattr(item, "prov") and item.prov:
            prov = item.prov
            if isinstance(prov, list) and len(prov) > 0:
                first_prov = prov[0]
                if hasattr(first_prov, "page_no"):
                    page = int(first_prov.page_no) if first_prov.page_no else 1
                elif hasattr(first_prov, "page"):
                    page = int(first_prov.page) if first_prov.page else 1
            elif hasattr(prov, "page_no"):
                page = int(prov.page_no) if prov.page_no else 1

        return ContentBlock(type=block_type, text=text, level=level, page=page)

    def to_llama_nodes(
        self,
        parsed: ParsedDocument,
        doc_id: str,
    ) -> list[TextNode]:
        """Convert a :class:`ParsedDocument` to a list of LlamaIndex TextNodes.

        **Critical**: the ``text`` field of every node is always
        ``f"{section_path}\\n{section_title}\\n{block.text}"``
        — never raw block text alone.

        Args:
            parsed: The parsed document output from :meth:`parse`.
            doc_id: Unique document identifier.

        Returns:
            List of :class:`TextNode` objects with full metadata.
        """
        nodes: list[TextNode] = []
        for i, block in enumerate(parsed.blocks):
            section_title = nearest_heading_above(parsed.blocks, i)
            section_path = build_section_path(parsed.blocks, i)
            toc_entry = nearest_toc_entry(parsed.toc, block)
            importance = compute_importance(block)

            # Heading depth: use block.level for headings, else depth of
            # nearest heading
            depth = block.level if block.type == "heading" else (
                toc_entry.level if toc_entry.section_id != "unknown" else 0
            )

            # CRITICAL: text always includes section context
            node_text = f"{section_path}\n{section_title}\n{block.text}"

            node = TextNode(
                text=node_text,
                id_=str(uuid.uuid4()),
                metadata={
                    "doc_id": doc_id,
                    "source_file": parsed.source_file,
                    "page": block.page,
                    "section_title": section_title,
                    "section_path": section_path,
                    "chunk_type": block.type,
                    "depth": depth,
                    "importance_score": importance,
                    "toc_section_id": toc_entry.section_id,
                    "parent_id": "",  # filled by HierarchicalNodeParser
                },
            )
            nodes.append(node)

        return nodes
