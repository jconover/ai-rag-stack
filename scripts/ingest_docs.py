#!/usr/bin/env python3
"""
DevOps Documentation Ingestion Pipeline
Processes markdown/text files and indexes them in Qdrant vector database

Features:
- Semantic markdown-aware chunking with heading preservation
- Code block protection (never split mid-block)
- Content type detection (prose, code, list, table)
- Metadata enrichment with heading paths
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector as QdrantSparseVector,
)
from tqdm import tqdm

# Import semantic chunker
from chunkers import (
    MarkdownSemanticChunker,
    ChunkConfig,
    ContentType,
    create_chunker_from_env,
    get_semantic_chunker,
)

# Configuration
DOCS_DIR = os.getenv("DOCS_DIR", "../data/docs")
CUSTOM_DOCS_DIR = os.getenv("CUSTOM_DOCS_DIR", "../data/custom")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "devops_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # Set to 'cuda' for GPU

# Chunking mode: 'semantic' (new) or 'legacy' (old RecursiveCharacterTextSplitter)
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "semantic")

# Hybrid search configuration
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
SPARSE_ENCODER_MODEL = os.getenv("SPARSE_ENCODER_MODEL", "Qdrant/bm25")

# Vector constants
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 384  # all-MiniLM-L6-v2


class DocumentIngestionPipeline:
    def __init__(self, use_semantic_chunking: bool = True, use_hybrid: bool = None):
        """
        Initialize the ingestion pipeline.

        Args:
            use_semantic_chunking: If True, use MarkdownSemanticChunker.
                                   If False, use legacy RecursiveCharacterTextSplitter.
            use_hybrid: If True, generate both dense and sparse vectors.
                       If None, read from HYBRID_SEARCH_ENABLED env var.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': EMBEDDING_DEVICE}
        )

        # Initialize chunker based on mode
        self.use_semantic_chunking = use_semantic_chunking

        if use_semantic_chunking:
            self.chunker = create_chunker_from_env()
            print(f"Using semantic chunking with config:")
            print(f"  - Prose: {self.chunker.config.prose_chunk_size} chars, {self.chunker.config.prose_chunk_overlap} overlap")
            print(f"  - Code: {self.chunker.config.code_chunk_size} chars, {self.chunker.config.code_chunk_overlap} overlap")
            print(f"  - Lists: {self.chunker.config.list_chunk_size} chars, {self.chunker.config.list_chunk_overlap} overlap")
            print(f"  - Tables: {self.chunker.config.table_chunk_size} chars, {self.chunker.config.table_chunk_overlap} overlap")
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
            print(f"Using legacy chunking: {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")

        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Hybrid search setup
        self.use_hybrid = use_hybrid if use_hybrid is not None else HYBRID_SEARCH_ENABLED
        self.sparse_encoder = None

        if self.use_hybrid:
            try:
                from fastembed import SparseTextEmbedding
                print(f"Initializing sparse encoder: {SPARSE_ENCODER_MODEL}")
                self.sparse_encoder = SparseTextEmbedding(model_name=SPARSE_ENCODER_MODEL)
                # Warmup
                list(self.sparse_encoder.embed(["warmup"]))
                print("Hybrid search enabled: generating both dense and sparse vectors")
            except ImportError:
                print("WARNING: fastembed not installed. Falling back to dense-only vectors.")
                print("  Install with: pip install fastembed")
                self.use_hybrid = False
            except Exception as e:
                print(f"WARNING: Failed to initialize sparse encoder: {e}")
                print("  Falling back to dense-only vectors.")
                self.use_hybrid = False

    def load_documents_from_directory(self, directory: str, source_name: str) -> List[Document]:
        """Load all markdown and text files from a directory"""
        documents = []

        # Load markdown files
        md_loader = DirectoryLoader(
            directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=MAX_WORKERS,
        )

        try:
            md_docs = md_loader.load()
            for doc in md_docs:
                doc.metadata['source_type'] = source_name
                doc.metadata['file_type'] = 'markdown'
            documents.extend(md_docs)
            print(f"Loaded {len(md_docs)} markdown files from {source_name}")
        except Exception as e:
            print(f"Error loading markdown from {directory}: {e}")

        # Load text files
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )

        try:
            txt_docs = txt_loader.load()
            for doc in txt_docs:
                doc.metadata['source_type'] = source_name
                doc.metadata['file_type'] = 'text'
            documents.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} text files from {source_name}")
        except Exception as e:
            print(f"Error loading text from {directory}: {e}")

        return documents

    def load_raw_markdown_files(self, directory: str, source_name: str) -> List[Document]:
        """
        Load markdown files with raw content preservation for semantic chunking.

        UnstructuredMarkdownLoader can lose markdown structure, so for semantic
        chunking we load files directly to preserve headers and formatting.
        """
        documents = []
        directory_path = Path(directory)

        if not directory_path.exists():
            return documents

        md_files = list(directory_path.rglob("*.md"))

        for md_file in tqdm(md_files, desc=f"Loading {source_name} markdown"):
            try:
                content = md_file.read_text(encoding='utf-8')

                # Create document with rich metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': str(md_file),
                        'source_type': source_name,
                        'file_type': 'markdown',
                        'file_name': md_file.name,
                        'relative_path': str(md_file.relative_to(directory_path)),
                    }
                )
                documents.append(doc)

            except Exception as e:
                print(f"Error loading {md_file}: {e}")

        print(f"Loaded {len(documents)} markdown files from {source_name}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using configured method.

        Args:
            documents: List of Document objects to split

        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []

        print(f"Splitting {len(documents)} documents into chunks...")

        if self.use_semantic_chunking:
            chunks = self.chunker.chunk_documents(documents)

            # Print statistics about chunk types
            content_types = {}
            for chunk in chunks:
                ct = chunk.metadata.get('content_type', 'unknown')
                content_types[ct] = content_types.get(ct, 0) + 1

            print(f"Created {len(chunks)} chunks:")
            for ct, count in sorted(content_types.items()):
                print(f"  - {ct}: {count} chunks")
        else:
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")

        return chunks

    def _ensure_hybrid_collection(self, collection_name: str):
        """Create or verify collection supports hybrid search (dense + sparse vectors)."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            print(f"Creating hybrid collection '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=DENSE_VECTOR_SIZE,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"Created collection '{collection_name}' with hybrid vector support")
        else:
            print(f"Collection '{collection_name}' already exists")

    def _ingest_hybrid(self, chunks: List[Document], collection_name: str, batch_size: int = 100):
        """Ingest documents with both dense and sparse vectors."""
        self._ensure_hybrid_collection(collection_name)

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Process in batches
        total_points = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc="Ingesting batches"):
            batch = chunks[i:i + batch_size]
            texts = [doc.page_content for doc in batch]

            # Generate dense embeddings
            dense_embeddings = self.embeddings.embed_documents(texts)

            # Generate sparse embeddings
            sparse_embeddings = list(self.sparse_encoder.embed(texts))

            # Create points with both vector types
            points = []
            for j, (doc, dense_vec, sparse_vec) in enumerate(zip(batch, dense_embeddings, sparse_embeddings)):
                point_id = total_points + j

                # Build payload from document metadata
                payload = {
                    'page_content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'source_type': doc.metadata.get('source_type', 'Unknown'),
                    **{k: v for k, v in doc.metadata.items()
                       if k not in ('page_content', 'source', 'source_type')}
                }

                point = PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: QdrantSparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        )
                    },
                    payload=payload,
                )
                points.append(point)

            # Upsert batch
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            total_points += len(batch)

        print(f"Successfully ingested {total_points} chunks with hybrid vectors")

    def ingest_documents(self, documents: List[Document], collection_name: str = COLLECTION_NAME):
        """Split documents and ingest into Qdrant"""
        if not documents:
            print("No documents to ingest")
            return

        # Split documents
        chunks = self.split_documents(documents)

        if not chunks:
            print("No chunks created from documents")
            return

        print(f"Ingesting into Qdrant collection '{collection_name}'...")

        # Use hybrid ingestion if enabled
        if self.use_hybrid and self.sparse_encoder is not None:
            self._ingest_hybrid(chunks, collection_name)
            return None  # Hybrid ingestion doesn't return a vectorstore

        # Create or update vector store (dense-only)
        vectorstore = Qdrant.from_documents(
            chunks,
            self.embeddings,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=collection_name,
            force_recreate=False,  # Set to True to recreate collection
        )

        print(f"Successfully ingested {len(chunks)} chunks into Qdrant")
        return vectorstore

    def run(self, use_raw_loading: bool = True):
        """
        Main ingestion pipeline

        Args:
            use_raw_loading: If True and using semantic chunking, load markdown files
                           directly to preserve structure. If False, use LangChain loaders.
        """
        all_documents = []

        # Documentation sources to ingest
        doc_sources = {
            "kubernetes": os.path.join(DOCS_DIR, "kubernetes"),
            "kubernetes-ai": os.path.join(DOCS_DIR, "kubernetes-ai"),
            "terraform": os.path.join(DOCS_DIR, "terraform"),
            "docker": os.path.join(DOCS_DIR, "docker"),
            "ansible": os.path.join(DOCS_DIR, "ansible"),
            "prometheus": os.path.join(DOCS_DIR, "prometheus"),
            "custom": CUSTOM_DOCS_DIR,
        }

        # Load documents from each source
        for source_name, directory in doc_sources.items():
            if os.path.exists(directory):
                print(f"\n{'='*60}")
                print(f"Processing {source_name} documentation...")
                print(f"{'='*60}")

                # Choose loading method based on chunking mode
                if self.use_semantic_chunking and use_raw_loading:
                    # Load raw markdown to preserve structure for semantic chunking
                    docs = self.load_raw_markdown_files(directory, source_name)
                else:
                    # Use LangChain loaders
                    docs = self.load_documents_from_directory(directory, source_name)

                all_documents.extend(docs)
            else:
                print(f"Directory not found: {directory} (skipping {source_name})")

        # Ingest all documents
        if all_documents:
            print(f"\n{'='*60}")
            print(f"Total documents loaded: {len(all_documents)}")
            print(f"{'='*60}\n")
            self.ingest_documents(all_documents)
        else:
            print("No documents found to ingest!")
            sys.exit(1)


def ingest_single_file(
    file_path: str,
    source_name: str = "custom",
    collection_name: str = COLLECTION_NAME,
    use_semantic_chunking: bool = True,
) -> int:
    """
    Ingest a single file into the vector database.

    Args:
        file_path: Path to the file to ingest
        source_name: Source type label for metadata
        collection_name: Qdrant collection name
        use_semantic_chunking: Whether to use semantic chunking

    Returns:
        Number of chunks created
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read file content
    content = file_path.read_text(encoding='utf-8')

    doc = Document(
        page_content=content,
        metadata={
            'source': str(file_path),
            'source_type': source_name,
            'file_type': 'markdown' if file_path.suffix == '.md' else 'text',
            'file_name': file_path.name,
        }
    )

    pipeline = DocumentIngestionPipeline(use_semantic_chunking=use_semantic_chunking)
    chunks = pipeline.split_documents([doc])

    if chunks:
        # Ingest chunks
        vectorstore = Qdrant.from_documents(
            chunks,
            pipeline.embeddings,
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=collection_name,
            force_recreate=False,
        )
        print(f"Ingested {len(chunks)} chunks from {file_path.name}")

    return len(chunks)


def main():
    # Determine chunking mode from environment
    use_semantic = CHUNKING_MODE.lower() == "semantic"

    print("Starting DevOps Documentation Ingestion Pipeline...")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunking mode: {'semantic' if use_semantic else 'legacy'}")
    print(f"Hybrid search: {'enabled' if HYBRID_SEARCH_ENABLED else 'disabled'}")

    if not use_semantic:
        print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    print()  # Blank line for readability

    pipeline = DocumentIngestionPipeline(use_semantic_chunking=use_semantic)
    pipeline.run(use_raw_loading=use_semantic)

    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
