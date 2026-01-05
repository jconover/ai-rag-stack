#!/usr/bin/env python3
"""
DevOps Documentation Ingestion Pipeline
Processes markdown/text files and indexes them in Qdrant vector database

Features:
- Semantic markdown-aware chunking with heading preservation
- Code block protection (never split mid-block)
- Content type detection (prose, code, list, table)
- Metadata enrichment with heading paths
- Incremental ingestion with change detection (only re-process changed files)

Usage:
    python ingest_docs.py              # Incremental ingestion (default)
    python ingest_docs.py --full       # Force full re-ingestion
    python ingest_docs.py --dry-run    # Show what would be processed
    python ingest_docs.py --stats      # Show registry statistics
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
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
    Filter,
    FieldCondition,
    MatchValue,
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

# Import ingestion registry for incremental updates
from ingestion_registry import (
    IngestionRegistry,
    ChangeSet,
    compute_file_hash,
    compute_config_hash,
    scan_directory_with_hashes,
    print_stats,
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

        # Initialize ingestion registry for incremental updates
        self.registry = IngestionRegistry()

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

    def _get_config_hash(self) -> str:
        """Get hash of current chunking configuration."""
        if self.use_semantic_chunking:
            # Use chunker config for semantic mode
            cfg = self.chunker.config
            config_str = f"semantic:{cfg.prose_chunk_size}:{cfg.prose_chunk_overlap}"
        else:
            config_str = f"legacy:{CHUNK_SIZE}:{CHUNK_OVERLAP}"
        return compute_config_hash(CHUNK_SIZE, CHUNK_OVERLAP, config_str)

    def _config_changed(self) -> bool:
        """Check if chunking configuration has changed since last ingestion."""
        current_hash = self._get_config_hash()
        stored_hash = self.registry.get_config_hash()

        if stored_hash is None:
            # First run, save config hash
            self.registry.set_config_hash(current_hash)
            return False

        if current_hash != stored_hash:
            print(f"WARNING: Chunking configuration has changed!")
            print(f"  Previous: {stored_hash}")
            print(f"  Current: {current_hash}")
            return True

        return False

    def delete_chunks_for_file(self, source_path: str, collection_name: str = COLLECTION_NAME) -> int:
        """
        Delete all chunks for a specific source file from Qdrant.

        Args:
            source_path: Path to the source file
            collection_name: Qdrant collection name

        Returns:
            Number of points deleted
        """
        try:
            # Count points before deletion
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_path),
                        )
                    ]
                ),
            )
            count = count_result.count

            if count > 0:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="source",
                                match=MatchValue(value=source_path),
                            )
                        ]
                    ),
                )

            return count
        except Exception as e:
            print(f"Error deleting chunks for {source_path}: {e}")
            return 0

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

    def run(self, use_raw_loading: bool = True, force_full: bool = False, dry_run: bool = False):
        """
        Main ingestion pipeline with incremental support.

        Args:
            use_raw_loading: If True and using semantic chunking, load markdown files
                           directly to preserve structure. If False, use LangChain loaders.
            force_full: If True, force full re-ingestion ignoring registry.
            dry_run: If True, only show what would be processed without making changes.
        """
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

        # Check if config changed (forces full re-ingestion)
        config_changed = self._config_changed()
        if config_changed and not force_full:
            print("\nChunking configuration changed. Recommend running with --full flag.")
            print("Continuing with incremental mode may result in inconsistent chunks.\n")

        if force_full:
            print("\n*** FULL RE-INGESTION MODE ***")
            print("All documents will be re-processed regardless of changes.\n")

        if dry_run:
            print("\n*** DRY RUN MODE ***")
            print("No changes will be made. Showing what would be processed.\n")

        # Collect all files with their hashes
        total_stats = {
            'new': 0, 'changed': 0, 'deleted': 0, 'unchanged': 0,
            'chunks_created': 0, 'chunks_deleted': 0
        }

        all_documents_to_process = []
        all_files_to_delete = []

        for source_name, directory in doc_sources.items():
            if not os.path.exists(directory):
                print(f"Directory not found: {directory} (skipping {source_name})")
                continue

            print(f"\n{'='*60}")
            print(f"Scanning {source_name} documentation...")
            print(f"{'='*60}")

            # Scan directory for all files with hashes
            current_files = scan_directory_with_hashes(
                Path(directory),
                extensions={'.md', '.txt', '.rst'},
            )

            if not current_files:
                print(f"No files found in {directory}")
                continue

            if force_full:
                # In full mode, treat all files as new
                new_files = list(current_files.keys())
                changed_files = []
                deleted_files = []
                unchanged_files = []
            else:
                # Detect changes using registry
                changes = self.registry.detect_changes(current_files, source_type=source_name)
                new_files = changes.new_files
                changed_files = changes.changed_files
                deleted_files = changes.deleted_files
                unchanged_files = changes.unchanged_files

            print(f"  Total files: {len(current_files)}")
            print(f"  New: {len(new_files)}, Changed: {len(changed_files)}, "
                  f"Deleted: {len(deleted_files)}, Unchanged: {len(unchanged_files)}")

            # Update stats
            total_stats['new'] += len(new_files)
            total_stats['changed'] += len(changed_files)
            total_stats['deleted'] += len(deleted_files)
            total_stats['unchanged'] += len(unchanged_files)

            # Collect files to process
            files_to_process = new_files + changed_files

            if files_to_process:
                for file_path in files_to_process:
                    try:
                        path = Path(file_path)
                        content = path.read_text(encoding='utf-8')

                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': file_path,
                                'source_type': source_name,
                                'file_type': 'markdown' if path.suffix == '.md' else 'text',
                                'file_name': path.name,
                                'relative_path': str(path.relative_to(Path(directory))),
                                'content_hash': current_files[file_path],
                                'file_size': path.stat().st_size,
                            }
                        )
                        all_documents_to_process.append(doc)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

            # Collect deleted files
            all_files_to_delete.extend(deleted_files)

            # For changed files, we need to delete old chunks first
            for file_path in changed_files:
                all_files_to_delete.append(file_path)

        # Summary
        print(f"\n{'='*60}")
        print("INGESTION SUMMARY")
        print(f"{'='*60}")
        print(f"Files to process: {len(all_documents_to_process)} "
              f"(new: {total_stats['new']}, changed: {total_stats['changed']})")
        print(f"Files to delete: {len(all_files_to_delete)}")
        print(f"Unchanged files: {total_stats['unchanged']}")

        if dry_run:
            if all_documents_to_process:
                print("\nFiles that would be processed:")
                for doc in all_documents_to_process[:20]:  # Show first 20
                    print(f"  + {doc.metadata['source']}")
                if len(all_documents_to_process) > 20:
                    print(f"  ... and {len(all_documents_to_process) - 20} more")

            if all_files_to_delete:
                print("\nFiles that would have chunks deleted:")
                for f in all_files_to_delete[:20]:
                    print(f"  - {f}")
                if len(all_files_to_delete) > 20:
                    print(f"  ... and {len(all_files_to_delete) - 20} more")

            print("\nDry run complete. No changes made.")
            return

        # Delete chunks for removed/changed files
        if all_files_to_delete:
            print(f"\nDeleting chunks for {len(all_files_to_delete)} files...")
            for file_path in tqdm(all_files_to_delete, desc="Deleting old chunks"):
                deleted_count = self.delete_chunks_for_file(file_path)
                total_stats['chunks_deleted'] += deleted_count

                # Remove from registry if file no longer exists
                if file_path not in [d.metadata['source'] for d in all_documents_to_process]:
                    self.registry.delete_file(file_path)

            print(f"Deleted {total_stats['chunks_deleted']} old chunks")

        # Process new/changed documents
        if all_documents_to_process:
            print(f"\nProcessing {len(all_documents_to_process)} documents...")
            chunks = self.split_documents(all_documents_to_process)

            if chunks:
                print(f"Created {len(chunks)} chunks")
                total_stats['chunks_created'] = len(chunks)

                # Ingest chunks
                self._ingest_chunks_with_registry(chunks)

                # Update config hash after successful ingestion
                self.registry.set_config_hash(self._get_config_hash())

        print(f"\n{'='*60}")
        print("INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"Chunks created: {total_stats['chunks_created']}")
        print(f"Chunks deleted: {total_stats['chunks_deleted']}")
        print_stats(self.registry)

    def _ingest_chunks_with_registry(self, chunks: List[Document], collection_name: str = COLLECTION_NAME):
        """
        Ingest chunks and update the registry with file tracking.

        Args:
            chunks: List of Document chunks to ingest
            collection_name: Qdrant collection name
        """
        if not chunks:
            return

        print(f"Ingesting {len(chunks)} chunks into Qdrant...")

        # Group chunks by source file for registry tracking
        chunks_by_source: Dict[str, List[Document]] = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)

        # Use hybrid ingestion if enabled
        if self.use_hybrid and self.sparse_encoder is not None:
            self._ingest_hybrid(chunks, collection_name)
        else:
            # Dense-only ingestion
            Qdrant.from_documents(
                chunks,
                self.embeddings,
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                collection_name=collection_name,
                force_recreate=False,
            )

        # Update registry for each file
        for source_path, source_chunks in chunks_by_source.items():
            if source_chunks:
                first_chunk = source_chunks[0]
                self.registry.update_file(
                    file_path=source_path,
                    content_hash=first_chunk.metadata.get('content_hash', ''),
                    source_type=first_chunk.metadata.get('source_type', 'unknown'),
                    chunk_count=len(source_chunks),
                    file_size=first_chunk.metadata.get('file_size', 0),
                )

        print(f"Successfully ingested {len(chunks)} chunks and updated registry")


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DevOps Documentation Ingestion Pipeline with incremental support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_docs.py              # Incremental ingestion (default)
  python ingest_docs.py --full       # Force full re-ingestion
  python ingest_docs.py --dry-run    # Preview what would be processed
  python ingest_docs.py --stats      # Show registry statistics only
  python ingest_docs.py --source kubernetes  # Process only kubernetes docs
        """
    )

    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Force full re-ingestion, ignoring registry (processes all files)'
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be processed without making changes'
    )

    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show registry statistics and exit'
    )

    parser.add_argument(
        '--source',
        type=str,
        help='Process only a specific source (e.g., kubernetes, terraform)'
    )

    parser.add_argument(
        '--clear-registry',
        action='store_true',
        help='Clear the ingestion registry (use with --full to re-index everything)'
    )

    parser.add_argument(
        '--legacy-chunking',
        action='store_true',
        help='Use legacy RecursiveCharacterTextSplitter instead of semantic chunking'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize registry for stats/clear operations
    registry = IngestionRegistry()

    # Handle stats-only mode
    if args.stats:
        print("Ingestion Registry Statistics")
        print_stats(registry)
        return

    # Handle clear registry
    if args.clear_registry:
        count = registry.clear()
        print(f"Cleared {count} entries from the ingestion registry")
        if not args.full:
            print("Tip: Use --full flag to re-index all documents")
        return

    # Determine chunking mode
    use_semantic = CHUNKING_MODE.lower() == "semantic" and not args.legacy_chunking

    print("Starting DevOps Documentation Ingestion Pipeline...")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunking mode: {'semantic' if use_semantic else 'legacy'}")
    print(f"Hybrid search: {'enabled' if HYBRID_SEARCH_ENABLED else 'disabled'}")
    print(f"Incremental mode: {'disabled (--full)' if args.full else 'enabled'}")

    if not use_semantic:
        print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    print()  # Blank line for readability

    pipeline = DocumentIngestionPipeline(use_semantic_chunking=use_semantic)
    pipeline.run(
        use_raw_loading=use_semantic,
        force_full=args.full,
        dry_run=args.dry_run,
    )

    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
