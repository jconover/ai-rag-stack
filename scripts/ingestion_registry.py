"""Ingestion Registry - SQLite-based tracking for incremental document ingestion.

This module provides persistent tracking of ingested documents to enable
incremental updates. Only changed files are re-processed on subsequent runs.

Usage:
    from ingestion_registry import IngestionRegistry

    registry = IngestionRegistry()

    # Check if file needs re-ingestion
    if registry.needs_update(file_path, current_hash):
        # Process file...
        registry.update_file(file_path, content_hash, source_type, chunk_count)

    # Get statistics
    stats = registry.get_stats()
"""
import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default registry location
DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "ingestion_registry.db"


@dataclass
class FileState:
    """State of an ingested file."""
    file_path: str
    content_hash: str
    source_type: str
    chunk_count: int
    file_size: int
    ingested_at: str
    updated_at: str
    chunk_ids: Optional[List[str]] = None  # List of deterministic UUID chunk IDs


@dataclass
class ChangeSet:
    """Results of change detection."""
    new_files: List[str]
    changed_files: List[str]
    deleted_files: List[str]
    unchanged_files: List[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.changed_files or self.deleted_files)

    @property
    def total_to_process(self) -> int:
        return len(self.new_files) + len(self.changed_files)

    def summary(self) -> str:
        return (
            f"New: {len(self.new_files)}, "
            f"Changed: {len(self.changed_files)}, "
            f"Deleted: {len(self.deleted_files)}, "
            f"Unchanged: {len(self.unchanged_files)}"
        )


class IngestionRegistry:
    """SQLite-based registry for tracking ingested documents.

    Provides:
    - Content hash tracking for change detection
    - Per-source-type statistics
    - Efficient lookups via indexed queries
    - ACID-compliant updates

    The registry file is stored at `data/ingestion_registry.db` by default.
    """

    SCHEMA_VERSION = 2  # Bumped for chunk_ids column

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the registry.

        Args:
            db_path: Path to SQLite database. Defaults to data/ingestion_registry.db
        """
        self.db_path = db_path or DEFAULT_REGISTRY_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Create main registry table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingested_files (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    ingested_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    chunk_ids TEXT
                )
            """)

            # Migration: add chunk_ids column if it doesn't exist
            try:
                conn.execute("SELECT chunk_ids FROM ingested_files LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE ingested_files ADD COLUMN chunk_ids TEXT")

            # Create indexes for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_type
                ON ingested_files(source_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash
                ON ingested_files(content_hash)
            """)

            # Create metadata table for config tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS registry_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Set schema version
            conn.execute("""
                INSERT OR REPLACE INTO registry_metadata (key, value)
                VALUES ('schema_version', ?)
            """, (str(self.SCHEMA_VERSION),))

            conn.commit()

    def get_file_state(self, file_path: str) -> Optional[FileState]:
        """Get the current state of a file from the registry.

        Args:
            file_path: Absolute or relative path to the file

        Returns:
            FileState if file is in registry, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM ingested_files WHERE file_path = ?",
                (file_path,)
            ).fetchone()

            if row:
                # Parse chunk_ids from JSON if present
                chunk_ids = None
                if row['chunk_ids']:
                    try:
                        chunk_ids = json.loads(row['chunk_ids'])
                    except json.JSONDecodeError:
                        chunk_ids = None

                return FileState(
                    file_path=row['file_path'],
                    content_hash=row['content_hash'],
                    source_type=row['source_type'],
                    chunk_count=row['chunk_count'],
                    file_size=row['file_size'],
                    ingested_at=row['ingested_at'],
                    updated_at=row['updated_at'],
                    chunk_ids=chunk_ids,
                )
            return None

    def needs_update(self, file_path: str, current_hash: str) -> bool:
        """Check if a file needs to be re-ingested.

        Args:
            file_path: Path to the file
            current_hash: Current content hash of the file

        Returns:
            True if file is new or has changed, False if unchanged
        """
        state = self.get_file_state(file_path)
        if state is None:
            return True  # New file
        return state.content_hash != current_hash

    def update_file(
        self,
        file_path: str,
        content_hash: str,
        source_type: str,
        chunk_count: int,
        file_size: int,
        chunk_ids: Optional[List[str]] = None,
    ) -> None:
        """Update or insert a file's state in the registry.

        Args:
            file_path: Path to the file
            content_hash: SHA-256 hash of file content
            source_type: Source type (kubernetes, terraform, etc.)
            chunk_count: Number of chunks created from this file
            file_size: File size in bytes
            chunk_ids: Optional list of deterministic UUID chunk IDs
        """
        now = datetime.utcnow().isoformat()
        chunk_ids_json = json.dumps(chunk_ids) if chunk_ids else None

        with self._get_connection() as conn:
            # Check if file exists
            existing = conn.execute(
                "SELECT ingested_at FROM ingested_files WHERE file_path = ?",
                (file_path,)
            ).fetchone()

            if existing:
                # Update existing
                conn.execute("""
                    UPDATE ingested_files
                    SET content_hash = ?, source_type = ?, chunk_count = ?,
                        file_size = ?, updated_at = ?, chunk_ids = ?
                    WHERE file_path = ?
                """, (content_hash, source_type, chunk_count, file_size, now, chunk_ids_json, file_path))
            else:
                # Insert new
                conn.execute("""
                    INSERT INTO ingested_files
                    (file_path, content_hash, source_type, chunk_count, file_size, ingested_at, updated_at, chunk_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (file_path, content_hash, source_type, chunk_count, file_size, now, now, chunk_ids_json))

            conn.commit()

    def delete_file(self, file_path: str) -> bool:
        """Remove a file from the registry.

        Args:
            file_path: Path to the file

        Returns:
            True if file was deleted, False if it didn't exist
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM ingested_files WHERE file_path = ?",
                (file_path,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_by_source_type(self, source_type: str) -> int:
        """Delete all files for a source type.

        Args:
            source_type: Source type to delete

        Returns:
            Number of files deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM ingested_files WHERE source_type = ?",
                (source_type,)
            )
            conn.commit()
            return cursor.rowcount

    def get_files_by_source(self, source_type: str) -> Dict[str, str]:
        """Get all files and their hashes for a source type.

        Args:
            source_type: Source type to query

        Returns:
            Dict mapping file_path -> content_hash
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT file_path, content_hash FROM ingested_files WHERE source_type = ?",
                (source_type,)
            ).fetchall()
            return {row['file_path']: row['content_hash'] for row in rows}

    def get_all_files(self) -> Dict[str, str]:
        """Get all files and their hashes.

        Returns:
            Dict mapping file_path -> content_hash
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT file_path, content_hash FROM ingested_files"
            ).fetchall()
            return {row['file_path']: row['content_hash'] for row in rows}

    def detect_changes(
        self,
        current_files: Dict[str, str],
        source_type: Optional[str] = None
    ) -> ChangeSet:
        """Detect changes between current files and registry.

        Args:
            current_files: Dict mapping file_path -> content_hash for current files
            source_type: Optional source type to limit comparison

        Returns:
            ChangeSet with categorized files
        """
        if source_type:
            registry_files = self.get_files_by_source(source_type)
        else:
            registry_files = self.get_all_files()

        current_paths = set(current_files.keys())
        registry_paths = set(registry_files.keys())

        new_files = []
        changed_files = []
        unchanged_files = []
        deleted_files = []

        # Find new and changed files
        for path in current_paths:
            if path not in registry_paths:
                new_files.append(path)
            elif current_files[path] != registry_files[path]:
                changed_files.append(path)
            else:
                unchanged_files.append(path)

        # Find deleted files
        for path in registry_paths:
            if path not in current_paths:
                deleted_files.append(path)

        return ChangeSet(
            new_files=sorted(new_files),
            changed_files=sorted(changed_files),
            deleted_files=sorted(deleted_files),
            unchanged_files=sorted(unchanged_files),
        )

    def get_stats(self) -> Dict[str, any]:
        """Get registry statistics.

        Returns:
            Dict with statistics by source type and totals
        """
        with self._get_connection() as conn:
            # Per-source stats
            rows = conn.execute("""
                SELECT source_type,
                       COUNT(*) as file_count,
                       SUM(chunk_count) as total_chunks,
                       SUM(file_size) as total_size
                FROM ingested_files
                GROUP BY source_type
                ORDER BY total_chunks DESC
            """).fetchall()

            by_source = {}
            for row in rows:
                by_source[row['source_type']] = {
                    'file_count': row['file_count'],
                    'chunk_count': row['total_chunks'] or 0,
                    'total_size': row['total_size'] or 0,
                }

            # Totals
            totals = conn.execute("""
                SELECT COUNT(*) as file_count,
                       SUM(chunk_count) as total_chunks,
                       SUM(file_size) as total_size
                FROM ingested_files
            """).fetchone()

            # Registry file size
            registry_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                'by_source': by_source,
                'total_files': totals['file_count'],
                'total_chunks': totals['total_chunks'] or 0,
                'total_size': totals['total_size'] or 0,
                'registry_size': registry_size,
                'registry_path': str(self.db_path),
            }

    def clear(self) -> int:
        """Clear all entries from the registry.

        Returns:
            Number of entries deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM ingested_files")
            conn.commit()
            return cursor.rowcount

    def set_config_hash(self, config_hash: str) -> None:
        """Store the chunking configuration hash.

        Used to detect when config changes require full re-ingestion.
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO registry_metadata (key, value)
                VALUES ('config_hash', ?)
            """, (config_hash,))
            conn.commit()

    def get_config_hash(self) -> Optional[str]:
        """Get the stored chunking configuration hash."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM registry_metadata WHERE key = 'config_hash'"
            ).fetchone()
            return row['value'] if row else None


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_config_hash(chunk_size: int, chunk_overlap: int, chunking_mode: str) -> str:
    """Compute hash of chunking configuration.

    Used to detect when config changes require full re-ingestion.
    """
    config_str = f"{chunk_size}:{chunk_overlap}:{chunking_mode}"
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def scan_directory_with_hashes(
    directory: Path,
    extensions: Set[str] = {'.md', '.txt', '.rst'},
    exclude_patterns: Set[str] = {'node_modules', '.git', '__pycache__', '.venv'},
) -> Dict[str, str]:
    """Scan a directory and compute hashes for all matching files.

    Args:
        directory: Directory to scan
        extensions: File extensions to include
        exclude_patterns: Directory names to exclude

    Returns:
        Dict mapping file_path -> content_hash
    """
    results = {}

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return results

    for root, dirs, files in os.walk(directory):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in exclude_patterns]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = Path(root) / file
                try:
                    file_hash = compute_file_hash(file_path)
                    results[str(file_path)] = file_hash
                except Exception as e:
                    logger.warning(f"Failed to hash {file_path}: {e}")

    return results


# Convenience function for CLI usage
def print_stats(registry: IngestionRegistry) -> None:
    """Print formatted registry statistics."""
    stats = registry.get_stats()

    print("\nIngestion Registry Statistics")
    print("=" * 50)
    print(f"Registry: {stats['registry_path']}")
    print(f"Registry size: {stats['registry_size'] / 1024:.1f} KB")
    print()

    print("By source type:")
    for source, data in sorted(stats['by_source'].items()):
        size_mb = data['total_size'] / (1024 * 1024)
        print(f"  {source:20} {data['file_count']:>6} files, {data['chunk_count']:>8} chunks, {size_mb:>6.1f} MB")

    print()
    print(f"Total: {stats['total_files']} files, {stats['total_chunks']} chunks")
