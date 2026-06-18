"""
ChromaDB Storage Layer
======================

Manages two separate ChromaDB collections:

* ``"chunks"`` — regular content chunks
* ``"toc_index"`` — dedicated TOC entry nodes

Each collection is wrapped in a :class:`ChromaVectorStore` and exposed
as a :class:`VectorStoreIndex`.  A :class:`SimpleDocumentStore` is also
maintained so that :class:`AutoMergingRetriever` can look up parent
nodes at retrieval time.
"""

from __future__ import annotations

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from embeddings.bge import BGEEmbedding


class ChromaStore:
    """Dual-collection ChromaDB store with document store for AutoMerging.

    Attributes:
        client: Persistent ChromaDB client.
        embed_model: BGE embedding model shared by both indices.
        chunks_collection: ChromaDB collection for content chunks.
        toc_collection: ChromaDB collection for TOC entries.
        storage_context: Storage context with docstore (set after
            :meth:`build_chunk_index` or :meth:`load_chunk_index`).

    Usage::

        store = ChromaStore()
        chunk_idx = store.build_chunk_index(chunked_nodes)
        toc_idx = store.build_toc_index(toc_nodes)
    """

    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )
        self.embed_model = BGEEmbedding()

        # --- Main chunks collection ---
        self.chunks_collection = self.client.get_or_create_collection("chunks")
        self.chunks_store = ChromaVectorStore(
            chroma_collection=self.chunks_collection,
        )

        # --- TOC index collection ---
        self.toc_collection = self.client.get_or_create_collection("toc_index")
        self.toc_store = ChromaVectorStore(
            chroma_collection=self.toc_collection,
        )

        # --- Document store for AutoMergingRetriever ---
        self.docstore = SimpleDocumentStore()

        # Set after building/loading the chunk index
        self.storage_context: StorageContext | None = None

    # ------------------------------------------------------------------
    # Build (index from scratch)
    # ------------------------------------------------------------------

    def build_chunk_index(self, nodes: list[TextNode]) -> VectorStoreIndex:
        """Build a fresh vector index from *nodes* and store in ChromaDB.

        Also populates the :attr:`docstore` so that
        :class:`AutoMergingRetriever` can look up parent nodes.

        Args:
            nodes: Chunked content :class:`TextNode` objects.

        Returns:
            A :class:`VectorStoreIndex` backed by the ``"chunks"``
            ChromaDB collection.
        """
        self.docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(
            vector_store=self.chunks_store,
            docstore=self.docstore,
        )
        self.storage_context = storage_context

        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

    def build_toc_index(self, toc_nodes: list[TextNode]) -> VectorStoreIndex:
        """Build a fresh vector index from TOC *toc_nodes*.

        Args:
            toc_nodes: TOC :class:`TextNode` objects from
                :class:`TOCIndexBuilder`.

        Returns:
            A :class:`VectorStoreIndex` backed by the ``"toc_index"``
            ChromaDB collection.
        """
        storage_context = StorageContext.from_defaults(
            vector_store=self.toc_store,
        )
        return VectorStoreIndex(
            toc_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

    # ------------------------------------------------------------------
    # Load (from existing persistent store)
    # ------------------------------------------------------------------

    def load_chunk_index(self) -> VectorStoreIndex:
        """Load the chunk index from the existing ChromaDB collection.

        Returns:
            A :class:`VectorStoreIndex` loaded from the persistent
            ``"chunks"`` collection.
        """
        storage_context = StorageContext.from_defaults(
            vector_store=self.chunks_store,
            docstore=self.docstore,
        )
        self.storage_context = storage_context

        return VectorStoreIndex.from_vector_store(
            self.chunks_store,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

    def load_toc_index(self) -> VectorStoreIndex:
        """Load the TOC index from the existing ChromaDB collection.

        Returns:
            A :class:`VectorStoreIndex` loaded from the persistent
            ``"toc_index"`` collection.
        """
        storage_context = StorageContext.from_defaults(
            vector_store=self.toc_store,
        )
        return VectorStoreIndex.from_vector_store(
            self.toc_store,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

    def load_all_chunk_nodes(self) -> list[TextNode]:
        """Load chunk nodes from the persistent ``"chunks"`` collection.

        Returns:
            List of :class:`TextNode` objects rebuilt from Chroma rows.
        """
        rows = self.chunks_collection.get(
            include=["documents", "metadatas"],
        )
        ids = rows.get("ids", []) or []
        docs = rows.get("documents", []) or []
        metadatas = rows.get("metadatas", []) or []

        nodes: list[TextNode] = []
        for idx, node_id in enumerate(ids):
            text = docs[idx] if idx < len(docs) and docs[idx] else ""
            metadata = (
                metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            )
            nodes.append(TextNode(text=text, metadata=metadata, id_=str(node_id)))
        return nodes
