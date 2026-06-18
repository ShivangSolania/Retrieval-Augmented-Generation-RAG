# Retrieval-Augmented Generation (RAG) Pipeline

A complete, production-ready RAG pipeline built with LlamaIndex, Docling, ChromaDB, Hugging Face BGE embeddings, and OpenAI. It supports robust document parsing, hybrid retrieval, cross-encoder reranking, and dynamic context expansion, accessible via a built-in FastAPI server or a command-line interface.

## Key Features

* **Advanced Document Parsing:** Uses Docling to parse PDFs (with OCR), DOCX, PPTX, HTML, Markdown, and plain text. Extracted tables, lists, and hierarchical headings build a navigable Table of Contents (TOC).
* **Audio Transcription:** Includes an audio-to-markdown transcription module using OpenAI's Whisper model (with optional speaker diarization).
* **Smart Chunking:** Employs intelligent routing to apply hierarchical or semantic chunking strategies based on document type and layout.
* **Hybrid Retrieval:** Combines dense vector search (Hugging Face `BAAI/bge-base-en-v1.5`) and sparse keyword search (BM25) via Reciprocal Rank Fusion (RRF).
* **Cross-Encoder Reranking:** Re-scores retrieved chunks for high-precision relevance using `BAAI/bge-reranker-base`.
* **Context Expansion:** Employs LlamaIndex's AutoMergingRetriever to expand high-density relevant child chunks into full parent contexts.
* **Dual Interface:** Run as a persistent REST API using FastAPI, or execute one-off commands using the CLI.

## Prerequisites

* Python 3.11 or higher
* OpenAI API Key (for GPT-4o generation and Whisper transcription)
* Minimum 4GB RAM & ~2GB Disk Space (for initial Hugging Face model downloads)

## Installation

1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root of the project with your OpenAI API key and any configuration overrides:

```env
OPENAI_API_KEY=sk-proj-...

# Optional configurations (defaults shown below)
CHUNK_SIZE=512
CHUNK_OVERLAP=64
PARENT_CHUNK_SIZE=1024
SEMANTIC_THRESHOLD=0.82
RETRIEVAL_TOP_K=50
RERANK_TOP_K=10
BM25_WEIGHT=0.4
VECTOR_WEIGHT=0.6
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
LLM_MODEL=gpt-4o
ENABLE_HYBRID=true
ENABLE_RERANKING=true
```

## Usage

You can interact with the pipeline either through the API server or via the CLI.

### Option 1: FastAPI Server

Start the API server:
```bash
uvicorn llamaindex_rag.api.main:app --host 0.0.0.0 --port 8000
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

**Common Endpoints:**
* **`POST /ingest`**: Process and index a document.
  ```bash
  curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"file_path\": \"/path/to/doc.pdf\", \"doc_id\": \"doc-001\"}"
  ```
* **`POST /search`**: Perform retrieval and reranking without generating an answer (useful for testing retrieval quality).
* **`POST /query`**: Full RAG pipeline (retrieve context and generate an LLM response).
  ```bash
  curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"What are the key features?\"}"
  ```

### Option 2: CLI Commands

Run the pipeline entirely from your terminal without starting a server:

```bash
# Ingest a file
python -m llamaindex_rag.cli ingest --file /path/to/doc.pdf --doc-id my-doc

# Query the pipeline
python -m llamaindex_rag.cli query "What are the key features?"

# Start an interactive Q&A session
python -m llamaindex_rag.cli interactive

# View collection diagnostics
python -m llamaindex_rag.cli health
```

### Audio Transcription

Convert meeting recordings or interviews to structured Markdown (which can then be ingested):

```bash
python llamaindex_rag/transcription/transcribe.py /path/to/audio.mp3
```

## Architecture Layout

* **`api/`**: FastAPI REST server setup and endpoint routing.
* **`chunking/`**: Node parsing and chunking strategy router.
* **`embeddings/`**: Wrapper for Hugging Face BGE embeddings.
* **`generation/`**: LLM generator interface for synthesizing responses.
* **`ingestion/`**: Docling-based parser and TOC index builder.
* **`observability/`**: Structured JSON logging.
* **`retrieval/`**: Hybrid retriever (BM25 + Vector), Cross-Encoder reranker, and Context Expander.
* **`storage/`**: ChromaDB dual-collection vector store configurations.
* **`transcription/`**: Whisper-based diarized audio transcription tools.
* **`pipeline.py`**: The main orchestrator connecting all RAG components.
