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
- Idempotent ingestion using content-based deterministic UUIDs (no duplicates on re-run)
- Chunk deduplication (exact and fuzzy) to remove duplicate content before embedding
- Transactional consistency with two-phase commit between registry and Qdrant

Usage:
    python ingest_docs.py              # Incremental ingestion (default)
    python ingest_docs.py --full       # Force full re-ingestion
    python ingest_docs.py --dry-run    # Show what would be processed
    python ingest_docs.py --stats      # Show registry statistics
    python ingest_docs.py --check-duplicates  # Check for duplicate vectors
    python ingest_docs.py --no-dedup   # Disable chunk deduplication
    python ingest_docs.py --fuzzy-dedup --fuzzy-threshold 0.9  # Enable fuzzy dedup
    python ingest_docs.py --cleanup-orphans  # Clean up orphaned staging entries

Environment Variables (Deduplication):
    DEDUPLICATION_ENABLED=true         # Enable/disable chunk deduplication (default: true)
    FUZZY_DEDUPLICATION_ENABLED=false  # Enable fuzzy near-duplicate detection (default: false)
    FUZZY_DEDUPLICATION_THRESHOLD=0.95 # Similarity threshold for fuzzy matching (default: 0.95)
"""

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

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

# Import chunk deduplication
from chunk_deduplication import (
    deduplicate_chunks,
    DeduplicationStats,
    log_deduplication_stats,
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
# Embedding model configuration
# Note: Changing the embedding model requires full re-ingestion (--full flag)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # Set to 'cuda' for GPU

# Chunking mode: 'semantic' (new) or 'legacy' (old RecursiveCharacterTextSplitter)
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "semantic")

# Hybrid search configuration
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
SPARSE_ENCODER_MODEL = os.getenv("SPARSE_ENCODER_MODEL", "Qdrant/bm25")

# Deduplication configuration
DEDUPLICATION_ENABLED = os.getenv("DEDUPLICATION_ENABLED", "true").lower() == "true"
FUZZY_DEDUPLICATION_ENABLED = os.getenv("FUZZY_DEDUPLICATION_ENABLED", "false").lower() == "true"
FUZZY_DEDUPLICATION_THRESHOLD = float(os.getenv("FUZZY_DEDUPLICATION_THRESHOLD", "0.95"))

# Vector constants
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# UUID namespace for deterministic chunk ID generation
# Using DNS namespace (standard UUID5 namespace) for consistency
CHUNK_ID_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


def generate_chunk_id(source_path: str, chunk_index: int, content: str) -> str:
    """Generate deterministic UUID based on content.

    Uses UUID5 with a namespace to ensure:
    - Same content always gets same ID (idempotent)
    - Different content gets different ID
    - IDs are valid UUIDs for Qdrant

    Args:
        source_path: Path to the source file
        chunk_index: Index of the chunk within the file
        content: The chunk's text content

    Returns:
        Deterministic UUID string
    """
    # Create a content hash for additional uniqueness
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
    # Combine source path, chunk index, and content hash
    identifier = f"{source_path}:{chunk_index}:{content_hash}"
    # Generate deterministic UUID5
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, identifier))


@dataclass
class StagedChunk:
    """Represents a chunk staged for ingestion."""
    chunk_id: str
    source_path: str
    content_hash: str
    metadata: Dict[str, Any]
    status: str = "staged"  # staged, committed, rolled_back
    staged_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class IngestionTransaction:
    """Two-phase commit transaction for document ingestion.

    Ensures transactional consistency between the ingestion registry (SQLite)
    and Qdrant vector database. Prevents orphaned chunks when Qdrant insertion
    fails after registry update.

    Usage:
        with IngestionTransaction(registry, qdrant_client) as txn:
            # Stage chunks before Qdrant insertion
            for chunk in chunks:
                txn.stage_chunk(chunk_id, source_path, content_hash, metadata)

            # Insert into Qdrant
            qdrant_client.upsert(points)

            # Mark transaction as committed after successful Qdrant upsert
            txn.commit()

        # If an exception occurs, rollback is automatic via __exit__
    """

    def __init__(self, registry: 'IngestionRegistry', qdrant_client: 'QdrantClient',
                 collection_name: str = COLLECTION_NAME):
        """Initialize the transaction.

        Args:
            registry: IngestionRegistry instance for SQLite operations
            qdrant_client: QdrantClient for vector database operations
            collection_name: Qdrant collection name for rollback operations
        """
        self.registry = registry
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.staged_chunks: List[StagedChunk] = []
        self.committed = False
        self.transaction_id = str(uuid.uuid4())
        self._init_staging_table()

    def _init_staging_table(self):
        """Initialize the staging table in SQLite if it doesn't exist."""
        with self.registry._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS staging_chunks (
                    transaction_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata TEXT,
                    status TEXT NOT NULL DEFAULT 'staged',
                    staged_at TEXT NOT NULL,
                    committed_at TEXT,
                    PRIMARY KEY (transaction_id, chunk_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_staging_status
                ON staging_chunks(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_staging_transaction
                ON staging_chunks(transaction_id)
            """)
            conn.commit()

    def __enter__(self) -> 'IngestionTransaction':
        """Enter the transaction context."""
        logger.debug(f"Starting ingestion transaction {self.transaction_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the transaction context, rolling back if not committed."""
        if exc_type is not None:
            # An exception occurred, perform rollback
            logger.warning(f"Exception in transaction {self.transaction_id}: {exc_val}")
            self.rollback()
        elif not self.committed:
            # No exception but not committed, still rollback
            logger.warning(f"Transaction {self.transaction_id} not committed, rolling back")
            self.rollback()
        return False  # Don't suppress exceptions

    def stage_chunk(self, chunk_id: str, source_path: str, content_hash: str,
                    metadata: Dict[str, Any]) -> StagedChunk:
        """Stage a chunk before Qdrant insertion.

        Args:
            chunk_id: Deterministic UUID for the chunk
            source_path: Path to the source file
            content_hash: Hash of the chunk content
            metadata: Chunk metadata dictionary

        Returns:
            StagedChunk instance
        """
        staged = StagedChunk(
            chunk_id=chunk_id,
            source_path=source_path,
            content_hash=content_hash,
            metadata=metadata,
            status="staged",
            staged_at=datetime.utcnow().isoformat(),
        )
        self.staged_chunks.append(staged)

        # Persist to staging table for recovery
        with self.registry._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO staging_chunks
                (transaction_id, chunk_id, source_path, content_hash, metadata, status, staged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                self.transaction_id,
                chunk_id,
                source_path,
                content_hash,
                json.dumps(metadata),
                "staged",
                staged.staged_at,
            ))
            conn.commit()

        return staged

    def stage_chunks_batch(self, chunks: List[Tuple[str, str, str, Dict[str, Any]]]) -> List[StagedChunk]:
        """Stage multiple chunks efficiently in a single transaction.

        Args:
            chunks: List of (chunk_id, source_path, content_hash, metadata) tuples

        Returns:
            List of StagedChunk instances
        """
        staged_list = []
        now = datetime.utcnow().isoformat()

        with self.registry._get_connection() as conn:
            for chunk_id, source_path, content_hash, metadata in chunks:
                staged = StagedChunk(
                    chunk_id=chunk_id,
                    source_path=source_path,
                    content_hash=content_hash,
                    metadata=metadata,
                    status="staged",
                    staged_at=now,
                )
                staged_list.append(staged)

                conn.execute("""
                    INSERT OR REPLACE INTO staging_chunks
                    (transaction_id, chunk_id, source_path, content_hash, metadata, status, staged_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.transaction_id,
                    chunk_id,
                    source_path,
                    content_hash,
                    json.dumps(metadata),
                    "staged",
                    now,
                ))
            conn.commit()

        self.staged_chunks.extend(staged_list)
        return staged_list

    def commit(self) -> bool:
        """Mark all staged chunks as committed after successful Qdrant upsert.

        This is the second phase of the two-phase commit. Call this only after
        Qdrant upsert succeeds.

        Returns:
            True if commit succeeded
        """
        if self.committed:
            logger.warning(f"Transaction {self.transaction_id} already committed")
            return True

        now = datetime.utcnow().isoformat()

        try:
            with self.registry._get_connection() as conn:
                # Update all staged chunks to committed
                conn.execute("""
                    UPDATE staging_chunks
                    SET status = 'committed', committed_at = ?
                    WHERE transaction_id = ? AND status = 'staged'
                """, (now, self.transaction_id))
                conn.commit()

            # Update in-memory state
            for chunk in self.staged_chunks:
                chunk.status = "committed"

            self.committed = True
            logger.info(f"Transaction {self.transaction_id} committed: {len(self.staged_chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to commit transaction {self.transaction_id}: {e}")
            return False

    def rollback(self) -> int:
        """Rollback staged chunks on failure.

        Removes staged entries from the staging table and attempts to delete
        any chunks that may have been inserted into Qdrant.

        Returns:
            Number of chunks rolled back
        """
        if self.committed:
            logger.warning(f"Cannot rollback committed transaction {self.transaction_id}")
            return 0

        rolled_back = 0

        try:
            # Get chunk IDs that were staged
            chunk_ids = [c.chunk_id for c in self.staged_chunks]

            if chunk_ids and self.qdrant_client is not None:
                # Attempt to delete from Qdrant (best effort)
                try:
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=chunk_ids,
                    )
                    logger.info(f"Deleted {len(chunk_ids)} chunks from Qdrant during rollback")
                except Exception as e:
                    logger.warning(f"Failed to delete chunks from Qdrant during rollback: {e}")

            # Update staging table
            with self.registry._get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE staging_chunks
                    SET status = 'rolled_back'
                    WHERE transaction_id = ? AND status = 'staged'
                """, (self.transaction_id,))
                rolled_back = cursor.rowcount
                conn.commit()

            # Update in-memory state
            for chunk in self.staged_chunks:
                if chunk.status == "staged":
                    chunk.status = "rolled_back"

            self.staged_chunks.clear()
            logger.info(f"Transaction {self.transaction_id} rolled back: {rolled_back} chunks")

        except Exception as e:
            logger.error(f"Error during rollback of transaction {self.transaction_id}: {e}")

        return rolled_back

    @staticmethod
    def cleanup_orphaned_entries(registry: 'IngestionRegistry',
                                  qdrant_client: 'QdrantClient',
                                  collection_name: str = COLLECTION_NAME,
                                  max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up orphaned staging entries from failed transactions.

        This should be run periodically or on startup to clean up entries from
        transactions that failed without proper rollback (e.g., process crash).

        Args:
            registry: IngestionRegistry instance
            qdrant_client: QdrantClient instance
            collection_name: Qdrant collection name
            max_age_hours: Maximum age in hours for staged entries before cleanup

        Returns:
            Dict with cleanup statistics
        """
        stats = {
            "orphaned_staged": 0,
            "rolled_back": 0,
            "deleted_from_qdrant": 0,
            "cleaned_committed": 0,
        }

        try:
            cutoff_time = datetime.utcnow()
            cutoff_iso = cutoff_time.isoformat()

            with registry._get_connection() as conn:
                # Find orphaned staged entries (older than max_age_hours)
                rows = conn.execute("""
                    SELECT transaction_id, chunk_id, staged_at
                    FROM staging_chunks
                    WHERE status = 'staged'
                    AND datetime(staged_at) < datetime(?, '-' || ? || ' hours')
                """, (cutoff_iso, max_age_hours)).fetchall()

                stats["orphaned_staged"] = len(rows)

                if rows:
                    # Group by transaction
                    orphaned_by_txn: Dict[str, List[str]] = {}
                    for row in rows:
                        txn_id = row['transaction_id']
                        if txn_id not in orphaned_by_txn:
                            orphaned_by_txn[txn_id] = []
                        orphaned_by_txn[txn_id].append(row['chunk_id'])

                    logger.info(f"Found {len(orphaned_by_txn)} orphaned transactions with "
                               f"{len(rows)} staged chunks")

                    # Attempt to delete from Qdrant and mark as rolled back
                    for txn_id, chunk_ids in orphaned_by_txn.items():
                        try:
                            # Delete from Qdrant
                            qdrant_client.delete(
                                collection_name=collection_name,
                                points_selector=chunk_ids,
                            )
                            stats["deleted_from_qdrant"] += len(chunk_ids)
                        except Exception as e:
                            logger.warning(f"Failed to delete orphaned chunks from Qdrant: {e}")

                        # Mark as rolled back
                        conn.execute("""
                            UPDATE staging_chunks
                            SET status = 'rolled_back'
                            WHERE transaction_id = ?
                        """, (txn_id,))
                        stats["rolled_back"] += len(chunk_ids)

                # Clean up old committed and rolled_back entries (keep for audit trail)
                # Only delete entries older than 7 days
                cursor = conn.execute("""
                    DELETE FROM staging_chunks
                    WHERE status IN ('committed', 'rolled_back')
                    AND datetime(staged_at) < datetime(?, '-7 days')
                """, (cutoff_iso,))
                stats["cleaned_committed"] = cursor.rowcount

                conn.commit()

        except Exception as e:
            logger.error(f"Error during orphan cleanup: {e}")

        return stats

    @staticmethod
    def get_staging_stats(registry: 'IngestionRegistry') -> Dict[str, Any]:
        """Get statistics about the staging table.

        Args:
            registry: IngestionRegistry instance

        Returns:
            Dict with staging table statistics
        """
        try:
            with registry._get_connection() as conn:
                # Count by status
                rows = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM staging_chunks
                    GROUP BY status
                """).fetchall()

                by_status = {row['status']: row['count'] for row in rows}

                # Count distinct transactions
                txn_count = conn.execute("""
                    SELECT COUNT(DISTINCT transaction_id) as count
                    FROM staging_chunks
                """).fetchone()['count']

                # Get oldest staged entry
                oldest = conn.execute("""
                    SELECT MIN(staged_at) as oldest
                    FROM staging_chunks
                    WHERE status = 'staged'
                """).fetchone()['oldest']

                return {
                    "by_status": by_status,
                    "total_transactions": txn_count,
                    "total_entries": sum(by_status.values()),
                    "staged_count": by_status.get("staged", 0),
                    "committed_count": by_status.get("committed", 0),
                    "rolled_back_count": by_status.get("rolled_back", 0),
                    "oldest_staged": oldest,
                }
        except Exception as e:
            logger.error(f"Error getting staging stats: {e}")
            return {
                "error": str(e),
                "by_status": {},
                "total_entries": 0,
            }


class DocumentIngestionPipeline:
    def __init__(
        self,
        use_semantic_chunking: bool = True,
        use_hybrid: bool = None,
        enable_deduplication: bool = None,
        enable_fuzzy_deduplication: bool = None,
        fuzzy_threshold: float = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            use_semantic_chunking: If True, use MarkdownSemanticChunker.
                                   If False, use legacy RecursiveCharacterTextSplitter.
            use_hybrid: If True, generate both dense and sparse vectors.
                       If None, read from HYBRID_SEARCH_ENABLED env var.
            enable_deduplication: If True, remove duplicate chunks before embedding.
                                 If None, read from DEDUPLICATION_ENABLED env var.
            enable_fuzzy_deduplication: If True, also remove near-duplicates.
                                       If None, read from FUZZY_DEDUPLICATION_ENABLED env var.
            fuzzy_threshold: Similarity threshold for fuzzy deduplication (0.0-1.0).
                            If None, read from FUZZY_DEDUPLICATION_THRESHOLD env var.
        """
        # Deduplication settings
        self.enable_deduplication = (
            enable_deduplication if enable_deduplication is not None
            else DEDUPLICATION_ENABLED
        )
        self.enable_fuzzy_deduplication = (
            enable_fuzzy_deduplication if enable_fuzzy_deduplication is not None
            else FUZZY_DEDUPLICATION_ENABLED
        )
        self.fuzzy_threshold = (
            fuzzy_threshold if fuzzy_threshold is not None
            else FUZZY_DEDUPLICATION_THRESHOLD
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBEDDING_DEVICE}
        )
        print(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION} dimensions)")

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
        """Get hash of current chunking and embedding configuration.

        Includes embedding model in the hash so that changing the model
        triggers a configuration change warning (requires full re-ingestion).
        """
        if self.use_semantic_chunking:
            # Use chunker config for semantic mode
            cfg = self.chunker.config
            config_str = f"semantic:{cfg.prose_chunk_size}:{cfg.prose_chunk_overlap}"
        else:
            config_str = f"legacy:{CHUNK_SIZE}:{CHUNK_OVERLAP}"
        # Include embedding model in config hash - changing model requires full re-ingestion
        config_str += f":embedding:{EMBEDDING_MODEL}:{EMBEDDING_DIMENSION}"
        return compute_config_hash(CHUNK_SIZE, CHUNK_OVERLAP, config_str)

    def _config_changed(self) -> bool:
        """Check if chunking or embedding configuration has changed since last ingestion."""
        current_hash = self._get_config_hash()
        stored_hash = self.registry.get_config_hash()

        if stored_hash is None:
            # First run, save config hash
            self.registry.set_config_hash(current_hash)
            return False

        if current_hash != stored_hash:
            print(f"WARNING: Configuration has changed (chunking or embedding model)!")
            print(f"  Previous hash: {stored_hash}")
            print(f"  Current hash: {current_hash}")
            print(f"  Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION} dims)")
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

    def check_duplicates(self, collection_name: str = COLLECTION_NAME, sample_size: int = 10000) -> Dict[str, any]:
        """
        Check for duplicate content in the vector store.

        Scans the collection for vectors with identical content hashes,
        which would indicate duplicates that should have been deduplicated
        by idempotent ingestion.

        Args:
            collection_name: Qdrant collection name
            sample_size: Maximum number of points to scan (for large collections)

        Returns:
            Dict with duplicate analysis results
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                return {
                    'error': f"Collection '{collection_name}' does not exist",
                    'total_points': 0,
                    'duplicates_found': 0,
                }

            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count

            print(f"Scanning collection '{collection_name}' ({total_points} points)...")

            # Scroll through points to check for duplicates
            content_hashes: Dict[str, List[str]] = {}  # hash -> list of point IDs
            points_scanned = 0
            offset = None

            while points_scanned < min(sample_size, total_points):
                # Fetch batch of points
                results, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not results:
                    break

                for point in results:
                    points_scanned += 1
                    content = point.payload.get('page_content', '')
                    source = point.payload.get('source', 'unknown')

                    # Create content hash for duplicate detection
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    key = f"{source}:{content_hash}"

                    if key not in content_hashes:
                        content_hashes[key] = []
                    content_hashes[key].append(str(point.id))

                if offset is None:
                    break

            # Find duplicates (same content from same source)
            duplicates = {k: v for k, v in content_hashes.items() if len(v) > 1}

            # Analyze results
            duplicate_count = sum(len(ids) - 1 for ids in duplicates.values())
            unique_content = len(content_hashes)

            result = {
                'total_points': total_points,
                'points_scanned': points_scanned,
                'unique_content': unique_content,
                'duplicate_groups': len(duplicates),
                'duplicate_points': duplicate_count,
                'duplication_rate': (duplicate_count / points_scanned * 100) if points_scanned > 0 else 0,
            }

            # Include sample duplicates for investigation
            if duplicates:
                sample_duplicates = []
                for key, ids in list(duplicates.items())[:5]:  # Show first 5 groups
                    source, _ = key.rsplit(':', 1)
                    sample_duplicates.append({
                        'source': source,
                        'point_ids': ids[:5],  # Limit to 5 IDs per group
                        'count': len(ids),
                    })
                result['sample_duplicates'] = sample_duplicates

            return result

        except Exception as e:
            return {
                'error': str(e),
                'total_points': 0,
                'duplicates_found': 0,
            }

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

    def split_documents(self, documents: List[Document]) -> Tuple[List[Document], Optional[DeduplicationStats]]:
        """
        Split documents into chunks using configured method, then deduplicate.

        Args:
            documents: List of Document objects to split

        Returns:
            Tuple of (chunked Document objects, deduplication stats or None)
        """
        if not documents:
            return [], None

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

        # Deduplicate chunks if enabled
        dedup_stats = None
        if self.enable_deduplication and chunks:
            print(f"\nDeduplicating {len(chunks)} chunks...")
            chunks, dedup_stats = deduplicate_chunks(
                chunks,
                enable_fuzzy=self.enable_fuzzy_deduplication,
                fuzzy_threshold=self.fuzzy_threshold,
                preserve_first=True,
                track_sources=True,
            )
            log_deduplication_stats(dedup_stats, verbose=True)

        return chunks, dedup_stats

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
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"Created collection '{collection_name}' with hybrid vector support "
                  f"(dense: {EMBEDDING_DIMENSION} dims)")
        else:
            print(f"Collection '{collection_name}' already exists")

    def _ingest_hybrid(self, chunks: List[Document], collection_name: str, batch_size: int = 100):
        """Ingest documents with both dense and sparse vectors using idempotent UUIDs.

        Uses content-based deterministic UUIDs to ensure:
        - Re-running ingestion updates existing vectors instead of creating duplicates
        - Same content always maps to same ID (idempotent)
        - Qdrant's upsert behavior handles update vs insert automatically
        """
        self._ensure_hybrid_collection(collection_name)

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Track chunk indices per source file for deterministic ID generation
        source_chunk_indices: Dict[str, int] = {}

        # Assign chunk indices and generate deterministic IDs
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in source_chunk_indices:
                source_chunk_indices[source] = 0
            chunk_index = source_chunk_indices[source]
            source_chunk_indices[source] += 1

            # Generate deterministic UUID and store in metadata
            chunk_id = generate_chunk_id(source, chunk_index, chunk.page_content)
            chunk.metadata['chunk_id'] = chunk_id
            chunk.metadata['chunk_index'] = chunk_index

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
            for doc, dense_vec, sparse_vec in zip(batch, dense_embeddings, sparse_embeddings):
                # Use deterministic UUID from metadata
                point_id = doc.metadata['chunk_id']

                # Build payload from document metadata
                payload = {
                    'page_content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'source_type': doc.metadata.get('source_type', 'Unknown'),
                    'chunk_id': point_id,
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    **{k: v for k, v in doc.metadata.items()
                       if k not in ('page_content', 'source', 'source_type', 'chunk_id', 'chunk_index')}
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

            # Upsert batch - same ID = update, different ID = insert
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            total_points += len(batch)

        print(f"Successfully ingested {total_points} chunks with hybrid vectors (idempotent UUIDs)")

    def ingest_documents(self, documents: List[Document], collection_name: str = COLLECTION_NAME):
        """Split documents and ingest into Qdrant using idempotent UUIDs.

        Uses content-based deterministic UUIDs to prevent duplicates on re-run.
        Same content always maps to same ID, enabling safe re-ingestion.
        Includes automatic chunk deduplication before embedding.
        """
        if not documents:
            print("No documents to ingest")
            return

        # Split and deduplicate documents
        chunks, dedup_stats = self.split_documents(documents)

        if not chunks:
            print("No chunks created from documents")
            return

        print(f"Ingesting into Qdrant collection '{collection_name}'...")

        # Use hybrid or dense-only ingestion with idempotent UUIDs
        if self.use_hybrid and self.sparse_encoder is not None:
            self._ingest_hybrid(chunks, collection_name)
        else:
            self._ingest_dense_only(chunks, collection_name)

        print(f"Successfully ingested {len(chunks)} chunks into Qdrant (idempotent)")
        return None  # Direct client operations don't return vectorstore

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
            # DevOps & Infrastructure
            "kubernetes": os.path.join(DOCS_DIR, "kubernetes"),
            "kubernetes-ai": os.path.join(DOCS_DIR, "kubernetes-ai"),
            "terraform": os.path.join(DOCS_DIR, "terraform"),
            "docker": os.path.join(DOCS_DIR, "docker"),
            "ansible": os.path.join(DOCS_DIR, "ansible"),
            "helm": os.path.join(DOCS_DIR, "helm"),
            "argocd": os.path.join(DOCS_DIR, "argocd"),
            "istio": os.path.join(DOCS_DIR, "istio"),
            # Monitoring & Logging
            "prometheus": os.path.join(DOCS_DIR, "prometheus"),
            "grafana": os.path.join(DOCS_DIR, "grafana"),
            "elasticsearch": os.path.join(DOCS_DIR, "elasticsearch"),
            "logstash": os.path.join(DOCS_DIR, "logstash"),
            "kibana": os.path.join(DOCS_DIR, "kibana"),
            # Programming Languages
            "python": os.path.join(DOCS_DIR, "python"),
            "go": os.path.join(DOCS_DIR, "go"),
            "rust": os.path.join(DOCS_DIR, "rust"),
            "nodejs": os.path.join(DOCS_DIR, "nodejs"),
            "javascript": os.path.join(DOCS_DIR, "javascript"),
            "bash": os.path.join(DOCS_DIR, "bash"),
            "zsh": os.path.join(DOCS_DIR, "zsh"),
            # CI/CD & GitOps
            "git": os.path.join(DOCS_DIR, "git"),
            "jenkins": os.path.join(DOCS_DIR, "jenkins"),
            "github-actions": os.path.join(DOCS_DIR, "github-actions"),
            "gitlab": os.path.join(DOCS_DIR, "gitlab"),
            # Cloud Platforms
            "aws": os.path.join(DOCS_DIR, "aws"),
            "azure": os.path.join(DOCS_DIR, "azure"),
            "gcp": os.path.join(DOCS_DIR, "gcp"),
            # Web & Networking
            "nginx": os.path.join(DOCS_DIR, "nginx"),
            "linux": os.path.join(DOCS_DIR, "linux"),
            # Automation & Integration
            "n8n": os.path.join(DOCS_DIR, "n8n"),
            # Configuration
            "config-formats": os.path.join(DOCS_DIR, "config-formats"),
            # Custom user docs
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
            chunks, dedup_stats = self.split_documents(all_documents_to_process)

            if chunks:
                print(f"Final chunk count after processing: {len(chunks)}")
                total_stats['chunks_created'] = len(chunks)

                # Track deduplication stats
                if dedup_stats:
                    total_stats['duplicates_removed'] = dedup_stats.total_removed

                # Ingest chunks
                self._ingest_chunks_with_registry(chunks)

                # Update config hash after successful ingestion
                self.registry.set_config_hash(self._get_config_hash())

        print(f"\n{'='*60}")
        print("INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"Chunks created: {total_stats['chunks_created']}")
        print(f"Chunks deleted: {total_stats['chunks_deleted']}")
        if 'duplicates_removed' in total_stats:
            print(f"Duplicates removed: {total_stats['duplicates_removed']}")
        print_stats(self.registry)

    def _ensure_dense_collection(self, collection_name: str):
        """Create or verify collection exists for dense-only vectors."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            print(f"Creating dense-only collection '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Created collection '{collection_name}' with dense vectors "
                  f"({EMBEDDING_DIMENSION} dims)")
        else:
            print(f"Collection '{collection_name}' already exists")

    def _ingest_dense_only(self, chunks: List[Document], collection_name: str, batch_size: int = 100):
        """Ingest documents with dense vectors only using idempotent UUIDs.

        Uses content-based deterministic UUIDs to ensure:
        - Re-running ingestion updates existing vectors instead of creating duplicates
        - Same content always maps to same ID (idempotent)
        - Qdrant's upsert behavior handles update vs insert automatically
        """
        self._ensure_dense_collection(collection_name)

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Track chunk indices per source file for deterministic ID generation
        source_chunk_indices: Dict[str, int] = {}

        # Assign chunk indices and generate deterministic IDs
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in source_chunk_indices:
                source_chunk_indices[source] = 0
            chunk_index = source_chunk_indices[source]
            source_chunk_indices[source] += 1

            # Generate deterministic UUID and store in metadata
            chunk_id = generate_chunk_id(source, chunk_index, chunk.page_content)
            chunk.metadata['chunk_id'] = chunk_id
            chunk.metadata['chunk_index'] = chunk_index

        # Process in batches
        total_points = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc="Ingesting batches"):
            batch = chunks[i:i + batch_size]
            texts = [doc.page_content for doc in batch]

            # Generate dense embeddings
            dense_embeddings = self.embeddings.embed_documents(texts)

            # Create points with dense vectors
            points = []
            for doc, dense_vec in zip(batch, dense_embeddings):
                # Use deterministic UUID from metadata
                point_id = doc.metadata['chunk_id']

                # Build payload from document metadata
                payload = {
                    'page_content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'source_type': doc.metadata.get('source_type', 'Unknown'),
                    'chunk_id': point_id,
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    **{k: v for k, v in doc.metadata.items()
                       if k not in ('page_content', 'source', 'source_type', 'chunk_id', 'chunk_index')}
                }

                point = PointStruct(
                    id=point_id,
                    vector=dense_vec,
                    payload=payload,
                )
                points.append(point)

            # Upsert batch - same ID = update, different ID = insert
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            total_points += len(batch)

        print(f"Successfully ingested {total_points} chunks with dense vectors (idempotent UUIDs)")

    def _ingest_chunks_with_registry(self, chunks: List[Document], collection_name: str = COLLECTION_NAME):
        """
        Ingest chunks and update the registry with file tracking using two-phase commit.

        Uses idempotent content-based UUIDs for all ingestion modes to prevent
        duplicate vectors on re-run. Same content always maps to same ID.

        Two-phase commit ensures:
        1. Chunks are staged in registry before Qdrant insertion
        2. Only marked as committed after successful Qdrant upsert
        3. Automatic rollback on failure (removes from Qdrant if needed)
        4. Orphaned chunks can be cleaned up on startup

        Args:
            chunks: List of Document chunks to ingest
            collection_name: Qdrant collection name
        """
        if not chunks:
            return

        print(f"Ingesting {len(chunks)} chunks into Qdrant with transactional consistency...")

        # Group chunks by source file for registry tracking
        chunks_by_source: Dict[str, List[Document]] = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)

        # Use two-phase commit for transactional consistency
        with IngestionTransaction(self.registry, self.client, collection_name) as txn:
            # Phase 1: Stage all chunks before Qdrant insertion
            print(f"Phase 1: Staging {len(chunks)} chunks in registry...")

            # Pre-compute chunk IDs for staging (same logic as _ingest_hybrid/_ingest_dense_only)
            source_chunk_indices: Dict[str, int] = {}
            staging_data = []

            for chunk in chunks:
                source = chunk.metadata.get('source', 'unknown')
                if source not in source_chunk_indices:
                    source_chunk_indices[source] = 0
                chunk_index = source_chunk_indices[source]
                source_chunk_indices[source] += 1

                # Generate deterministic UUID
                chunk_id = generate_chunk_id(source, chunk_index, chunk.page_content)
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()

                # Store ID in chunk metadata for later use
                chunk.metadata['chunk_id'] = chunk_id
                chunk.metadata['chunk_index'] = chunk_index

                # Prepare staging data
                staging_data.append((
                    chunk_id,
                    source,
                    content_hash,
                    {
                        'source_type': chunk.metadata.get('source_type', 'unknown'),
                        'chunk_index': chunk_index,
                        'file_size': chunk.metadata.get('file_size', 0),
                    }
                ))

            # Batch stage all chunks
            txn.stage_chunks_batch(staging_data)
            print(f"Staged {len(staging_data)} chunks in transaction {txn.transaction_id}")

            # Phase 2: Insert into Qdrant (using existing methods but skipping ID generation)
            print(f"Phase 2: Inserting into Qdrant...")
            if self.use_hybrid and self.sparse_encoder is not None:
                self._ingest_hybrid_transactional(chunks, collection_name)
            else:
                self._ingest_dense_only_transactional(chunks, collection_name)

            # Phase 3: Commit transaction after successful Qdrant insertion
            print(f"Phase 3: Committing transaction...")
            if not txn.commit():
                raise RuntimeError("Failed to commit ingestion transaction")

        # Update main registry for each file with chunk IDs (after successful commit)
        for source_path, source_chunks in chunks_by_source.items():
            if source_chunks:
                first_chunk = source_chunks[0]
                chunk_ids = [c.metadata.get('chunk_id') for c in source_chunks if c.metadata.get('chunk_id')]
                self.registry.update_file(
                    file_path=source_path,
                    content_hash=first_chunk.metadata.get('content_hash', ''),
                    source_type=first_chunk.metadata.get('source_type', 'unknown'),
                    chunk_count=len(source_chunks),
                    file_size=first_chunk.metadata.get('file_size', 0),
                    chunk_ids=chunk_ids if chunk_ids else None,
                )

        print(f"Successfully ingested {len(chunks)} chunks with transactional consistency")

    def _ingest_hybrid_transactional(self, chunks: List[Document], collection_name: str, batch_size: int = 100):
        """Ingest documents with both dense and sparse vectors (transactional version).

        Assumes chunk IDs are already assigned in metadata (by _ingest_chunks_with_registry).
        """
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
            for doc, dense_vec, sparse_vec in zip(batch, dense_embeddings, sparse_embeddings):
                # Use pre-assigned deterministic UUID from metadata
                point_id = doc.metadata['chunk_id']

                # Build payload from document metadata
                payload = {
                    'page_content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'source_type': doc.metadata.get('source_type', 'Unknown'),
                    'chunk_id': point_id,
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    **{k: v for k, v in doc.metadata.items()
                       if k not in ('page_content', 'source', 'source_type', 'chunk_id', 'chunk_index')}
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

        print(f"Inserted {total_points} chunks with hybrid vectors")

    def _ingest_dense_only_transactional(self, chunks: List[Document], collection_name: str, batch_size: int = 100):
        """Ingest documents with dense vectors only (transactional version).

        Assumes chunk IDs are already assigned in metadata (by _ingest_chunks_with_registry).
        """
        self._ensure_dense_collection(collection_name)

        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Process in batches
        total_points = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc="Ingesting batches"):
            batch = chunks[i:i + batch_size]
            texts = [doc.page_content for doc in batch]

            # Generate dense embeddings
            dense_embeddings = self.embeddings.embed_documents(texts)

            # Create points with dense vectors
            points = []
            for doc, dense_vec in zip(batch, dense_embeddings):
                # Use pre-assigned deterministic UUID from metadata
                point_id = doc.metadata['chunk_id']

                # Build payload from document metadata
                payload = {
                    'page_content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'source_type': doc.metadata.get('source_type', 'Unknown'),
                    'chunk_id': point_id,
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    **{k: v for k, v in doc.metadata.items()
                       if k not in ('page_content', 'source', 'source_type', 'chunk_id', 'chunk_index')}
                }

                point = PointStruct(
                    id=point_id,
                    vector=dense_vec,
                    payload=payload,
                )
                points.append(point)

            # Upsert batch
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            total_points += len(batch)

        print(f"Inserted {total_points} chunks with dense vectors")


def ingest_single_file(
    file_path: str,
    source_name: str = "custom",
    collection_name: str = COLLECTION_NAME,
    use_semantic_chunking: bool = True,
    enable_deduplication: bool = True,
) -> int:
    """
    Ingest a single file into the vector database using idempotent UUIDs.

    Uses content-based deterministic UUIDs to ensure:
    - Re-running ingestion updates existing vectors instead of creating duplicates
    - Same content always maps to same ID (idempotent)
    - Automatic chunk deduplication before embedding

    Args:
        file_path: Path to the file to ingest
        source_name: Source type label for metadata
        collection_name: Qdrant collection name
        use_semantic_chunking: Whether to use semantic chunking
        enable_deduplication: Whether to deduplicate chunks

    Returns:
        Number of chunks created (after deduplication)
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

    pipeline = DocumentIngestionPipeline(
        use_semantic_chunking=use_semantic_chunking,
        enable_deduplication=enable_deduplication,
    )
    chunks, dedup_stats = pipeline.split_documents([doc])

    if chunks:
        # Use idempotent ingestion with deterministic UUIDs
        if pipeline.use_hybrid and pipeline.sparse_encoder is not None:
            pipeline._ingest_hybrid(chunks, collection_name)
        else:
            pipeline._ingest_dense_only(chunks, collection_name)

        dedup_msg = ""
        if dedup_stats and dedup_stats.total_removed > 0:
            dedup_msg = f", {dedup_stats.total_removed} duplicates removed"
        print(f"Ingested {len(chunks)} chunks from {file_path.name} (idempotent{dedup_msg})")

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
  python ingest_docs.py --check-duplicates   # Check for duplicate vectors
  python ingest_docs.py --no-dedup           # Disable chunk deduplication
  python ingest_docs.py --fuzzy-dedup        # Enable fuzzy near-duplicate removal
  python ingest_docs.py --fuzzy-dedup --fuzzy-threshold 0.9  # Custom threshold
  python ingest_docs.py --cleanup-orphans    # Clean up orphaned staging entries
  python ingest_docs.py --staging-stats      # Show staging table statistics

Note: This script uses idempotent content-based UUIDs. Re-running ingestion
will update existing vectors instead of creating duplicates. Safe to run
'make ingest' multiple times.

Transactional Consistency:
This script uses two-phase commit between the registry and Qdrant to prevent
orphaned chunks. Chunks are staged in SQLite before Qdrant insertion, and
only marked as committed after successful upsert. On failure, automatic
rollback removes any partially inserted chunks from Qdrant.

Run --cleanup-orphans periodically or on startup to clean up entries from
transactions that failed without proper rollback (e.g., process crash).

Deduplication removes duplicate chunks BEFORE embedding, saving compute costs
and improving search quality. Exact deduplication is enabled by default.
Fuzzy deduplication can catch near-duplicates with minor differences.
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

    parser.add_argument(
        '--check-duplicates',
        action='store_true',
        help='Check for duplicate content in the vector store and report findings'
    )

    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable chunk deduplication during ingestion'
    )

    parser.add_argument(
        '--fuzzy-dedup',
        action='store_true',
        help='Enable fuzzy (near-duplicate) deduplication in addition to exact matching'
    )

    parser.add_argument(
        '--fuzzy-threshold',
        type=float,
        default=0.95,
        help='Similarity threshold for fuzzy deduplication (0.0-1.0, default: 0.95)'
    )

    parser.add_argument(
        '--cleanup-orphans',
        action='store_true',
        help='Clean up orphaned staging entries from failed transactions'
    )

    parser.add_argument(
        '--staging-stats',
        action='store_true',
        help='Show staging table statistics (for transaction debugging)'
    )

    parser.add_argument(
        '--orphan-max-age',
        type=int,
        default=24,
        help='Maximum age in hours for staged entries before cleanup (default: 24)'
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

    # Handle duplicate check mode
    if args.check_duplicates:
        print("Checking for duplicate content in vector store...")
        print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"Collection: {COLLECTION_NAME}")
        print()

        # Create minimal pipeline just for Qdrant client (skip embedding model)
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Create a temporary pipeline instance for the check_duplicates method
        class DuplicateChecker:
            def __init__(self, qdrant_client):
                self.client = qdrant_client

        checker = DuplicateChecker(client)
        # Borrow the method
        checker.check_duplicates = DocumentIngestionPipeline.check_duplicates.__get__(checker, DuplicateChecker)

        result = checker.check_duplicates(COLLECTION_NAME)

        if 'error' in result:
            print(f"Error: {result['error']}")
            return

        print(f"{'='*60}")
        print("DUPLICATE CHECK RESULTS")
        print(f"{'='*60}")
        print(f"Total points in collection: {result['total_points']}")
        print(f"Points scanned: {result['points_scanned']}")
        print(f"Unique content chunks: {result['unique_content']}")
        print(f"Duplicate groups found: {result['duplicate_groups']}")
        print(f"Duplicate points (extras): {result['duplicate_points']}")
        print(f"Duplication rate: {result['duplication_rate']:.2f}%")

        if result.get('sample_duplicates'):
            print(f"\nSample duplicate groups:")
            for dup in result['sample_duplicates']:
                print(f"  - Source: {dup['source']}")
                print(f"    Count: {dup['count']} duplicates")
                print(f"    Point IDs: {', '.join(dup['point_ids'][:3])}...")

        if result['duplicate_points'] > 0:
            print(f"\nRecommendation: Run 'python ingest_docs.py --full' to re-ingest")
            print("with idempotent UUIDs, which will deduplicate the collection.")
        else:
            print("\nNo duplicates found. Collection is clean.")
        return

    # Handle staging stats mode
    if args.staging_stats:
        print("Staging Table Statistics")
        print(f"{'='*60}")

        stats = IngestionTransaction.get_staging_stats(registry)

        if 'error' in stats:
            print(f"Error: {stats['error']}")
            return

        print(f"Total entries: {stats['total_entries']}")
        print(f"Total transactions: {stats['total_transactions']}")
        print()
        print("By status:")
        for status, count in stats.get('by_status', {}).items():
            print(f"  {status}: {count}")
        print()
        print(f"Staged (pending): {stats['staged_count']}")
        print(f"Committed: {stats['committed_count']}")
        print(f"Rolled back: {stats['rolled_back_count']}")

        if stats.get('oldest_staged'):
            print(f"\nOldest pending staged entry: {stats['oldest_staged']}")
            print("Consider running --cleanup-orphans if these are old entries")
        else:
            print("\nNo pending staged entries")
        return

    # Handle orphan cleanup mode
    if args.cleanup_orphans:
        print("Cleaning up orphaned staging entries...")
        print(f"{'='*60}")
        print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Max age for orphans: {args.orphan_max_age} hours")
        print()

        # Create Qdrant client
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Run cleanup
        cleanup_stats = IngestionTransaction.cleanup_orphaned_entries(
            registry=registry,
            qdrant_client=client,
            collection_name=COLLECTION_NAME,
            max_age_hours=args.orphan_max_age,
        )

        print(f"{'='*60}")
        print("CLEANUP RESULTS")
        print(f"{'='*60}")
        print(f"Orphaned staged entries found: {cleanup_stats['orphaned_staged']}")
        print(f"Entries rolled back: {cleanup_stats['rolled_back']}")
        print(f"Chunks deleted from Qdrant: {cleanup_stats['deleted_from_qdrant']}")
        print(f"Old committed/rolled_back entries cleaned: {cleanup_stats['cleaned_committed']}")

        if cleanup_stats['orphaned_staged'] > 0:
            print("\nOrphaned entries have been cleaned up.")
        else:
            print("\nNo orphaned entries found. System is clean.")
        return

    # Determine chunking mode
    use_semantic = CHUNKING_MODE.lower() == "semantic" and not args.legacy_chunking

    # Determine deduplication settings from args
    enable_dedup = not args.no_dedup
    enable_fuzzy = args.fuzzy_dedup
    fuzzy_threshold = args.fuzzy_threshold

    print("Starting DevOps Documentation Ingestion Pipeline...")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION} dims, device={EMBEDDING_DEVICE})")
    print(f"Chunking mode: {'semantic' if use_semantic else 'legacy'}")
    print(f"Hybrid search: {'enabled' if HYBRID_SEARCH_ENABLED else 'disabled'}")
    print(f"Incremental mode: {'disabled (--full)' if args.full else 'enabled'}")
    print(f"Idempotent UUIDs: enabled (content-based deterministic IDs)")
    print(f"Transactional consistency: enabled (two-phase commit with rollback)")

    # Deduplication status
    if enable_dedup:
        dedup_mode = "exact"
        if enable_fuzzy:
            dedup_mode = f"exact + fuzzy (threshold={fuzzy_threshold})"
        print(f"Deduplication: enabled ({dedup_mode})")
    else:
        print(f"Deduplication: disabled")

    if not use_semantic:
        print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    print()  # Blank line for readability

    pipeline = DocumentIngestionPipeline(
        use_semantic_chunking=use_semantic,
        enable_deduplication=enable_dedup,
        enable_fuzzy_deduplication=enable_fuzzy,
        fuzzy_threshold=fuzzy_threshold,
    )
    pipeline.run(
        use_raw_loading=use_semantic,
        force_full=args.full,
        dry_run=args.dry_run,
    )

    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
