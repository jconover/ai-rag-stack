#!/usr/bin/env python3
"""
Chunk Deduplication Module for DevOps Documentation Ingestion

Provides exact and fuzzy deduplication of document chunks to:
- Remove identical chunks (same content from different sources or overlapping ingestions)
- Optionally identify near-duplicate chunks using similarity hashing
- Track and log deduplication statistics

Usage:
    from chunk_deduplication import deduplicate_chunks, DeduplicationStats

    chunks, stats = deduplicate_chunks(chunks, fuzzy_threshold=0.9)
    print(f"Removed {stats.exact_duplicates} exact duplicates")
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema.document import Document
    except ImportError:
        try:
            from langchain.schema import Document
        except ImportError:
            # Fallback for standalone testing without langchain
            class Document:
                """Simple Document mock for testing without langchain."""
                def __init__(self, page_content: str = "", metadata: dict = None):
                    self.page_content = page_content
                    self.metadata = metadata or {}


@dataclass
class DeduplicationStats:
    """Statistics from chunk deduplication process"""
    total_input_chunks: int = 0
    total_output_chunks: int = 0
    exact_duplicates_removed: int = 0
    fuzzy_duplicates_removed: int = 0
    duplicates_by_source: Dict[str, int] = field(default_factory=dict)

    @property
    def total_removed(self) -> int:
        return self.exact_duplicates_removed + self.fuzzy_duplicates_removed

    @property
    def deduplication_rate(self) -> float:
        if self.total_input_chunks == 0:
            return 0.0
        return (self.total_removed / self.total_input_chunks) * 100

    def __str__(self) -> str:
        return (
            f"Deduplication Stats:\n"
            f"  Input chunks: {self.total_input_chunks}\n"
            f"  Output chunks: {self.total_output_chunks}\n"
            f"  Exact duplicates removed: {self.exact_duplicates_removed}\n"
            f"  Fuzzy duplicates removed: {self.fuzzy_duplicates_removed}\n"
            f"  Total removed: {self.total_removed} ({self.deduplication_rate:.1f}%)"
        )


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent hashing.

    Normalization steps:
    - Convert to lowercase
    - Normalize unicode characters
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    - Remove common boilerplate patterns

    Args:
        text: Raw text content

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    # Unicode normalization (NFKC: compatibility decomposition + canonical composition)
    text = unicodedata.normalize('NFKC', text)

    # Convert to lowercase
    text = text.lower()

    # Collapse multiple whitespace (including newlines) to single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def compute_chunk_hash(content: str, normalize: bool = True) -> str:
    """
    Compute a deterministic hash for chunk content.

    Uses SHA-256 for collision resistance. Optional normalization
    ensures that chunks with minor formatting differences hash to
    the same value.

    Args:
        content: The chunk text content
        normalize: If True, normalize text before hashing (recommended)

    Returns:
        Hex string of SHA-256 hash (64 characters)
    """
    if normalize:
        content = normalize_text(content)

    # Use SHA-256 for good collision resistance
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def compute_simhash(text: str, hash_bits: int = 64) -> int:
    """
    Compute SimHash for near-duplicate detection.

    SimHash is a locality-sensitive hash that produces similar
    hashes for similar documents. Documents with small Hamming
    distance between their SimHashes are likely near-duplicates.

    Args:
        text: The text content to hash
        hash_bits: Number of bits in the hash (default 64)

    Returns:
        Integer SimHash value
    """
    if not text:
        return 0

    # Normalize and tokenize
    normalized = normalize_text(text)
    # Simple tokenization: split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', normalized)

    if not tokens:
        return 0

    # Initialize bit counts
    bit_counts = [0] * hash_bits

    # For each token, compute hash and update bit counts
    for token in tokens:
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)

        for i in range(hash_bits):
            bit_value = (token_hash >> i) & 1
            if bit_value:
                bit_counts[i] += 1
            else:
                bit_counts[i] -= 1

    # Generate final hash: bit is 1 if count > 0
    simhash = 0
    for i in range(hash_bits):
        if bit_counts[i] > 0:
            simhash |= (1 << i)

    return simhash


def hamming_distance(hash1: int, hash2: int, hash_bits: int = 64) -> int:
    """
    Compute Hamming distance between two hashes.

    Args:
        hash1: First hash value
        hash2: Second hash value
        hash_bits: Number of bits in the hash

    Returns:
        Number of differing bits
    """
    xor = hash1 ^ hash2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance


def simhash_similarity(hash1: int, hash2: int, hash_bits: int = 64) -> float:
    """
    Compute similarity score between two SimHashes.

    Args:
        hash1: First SimHash value
        hash2: Second SimHash value
        hash_bits: Number of bits in the hash

    Returns:
        Similarity score between 0.0 and 1.0
    """
    distance = hamming_distance(hash1, hash2, hash_bits)
    return 1.0 - (distance / hash_bits)


def deduplicate_chunks(
    chunks: List[Document],
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 0.95,
    preserve_first: bool = True,
    track_sources: bool = True,
) -> Tuple[List[Document], DeduplicationStats]:
    """
    Remove duplicate chunks from a list of documents.

    Performs exact deduplication by default, with optional fuzzy
    deduplication for near-duplicates. When duplicates are found,
    keeps the first occurrence by default.

    Args:
        chunks: List of Document objects to deduplicate
        enable_fuzzy: If True, also remove near-duplicates using SimHash
        fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
                        Only used if enable_fuzzy=True
        preserve_first: If True, keep first occurrence; if False, keep last
        track_sources: If True, track which sources had duplicates

    Returns:
        Tuple of (deduplicated_chunks, statistics)
    """
    stats = DeduplicationStats(total_input_chunks=len(chunks))

    if not chunks:
        return [], stats

    # Phase 1: Exact deduplication using content hash
    seen_hashes: Dict[str, int] = {}  # hash -> index of first occurrence
    exact_duplicate_indices: Set[int] = set()

    for i, chunk in enumerate(chunks):
        content_hash = compute_chunk_hash(chunk.page_content)

        if content_hash in seen_hashes:
            # This is a duplicate
            if preserve_first:
                exact_duplicate_indices.add(i)
            else:
                exact_duplicate_indices.add(seen_hashes[content_hash])
                seen_hashes[content_hash] = i

            # Track source for statistics
            if track_sources:
                source = chunk.metadata.get('source', 'unknown')
                source_type = chunk.metadata.get('source_type', 'unknown')
                key = f"{source_type}:{source}"
                stats.duplicates_by_source[key] = stats.duplicates_by_source.get(key, 0) + 1
        else:
            seen_hashes[content_hash] = i

    stats.exact_duplicates_removed = len(exact_duplicate_indices)

    # Build list of chunks after exact deduplication
    after_exact = [
        (i, chunk) for i, chunk in enumerate(chunks)
        if i not in exact_duplicate_indices
    ]

    # Phase 2: Fuzzy deduplication (optional)
    fuzzy_duplicate_indices: Set[int] = set()

    if enable_fuzzy and fuzzy_threshold < 1.0:
        # Compute SimHashes for remaining chunks
        simhashes: List[Tuple[int, int, Document]] = []  # (original_index, simhash, chunk)

        for orig_idx, chunk in after_exact:
            sh = compute_simhash(chunk.page_content)
            simhashes.append((orig_idx, sh, chunk))

        # Compare pairs to find near-duplicates
        # This is O(n^2) but necessary for fuzzy matching
        # For large datasets, consider using LSH buckets
        n = len(simhashes)

        for i in range(n):
            if simhashes[i][0] in fuzzy_duplicate_indices:
                continue

            for j in range(i + 1, n):
                if simhashes[j][0] in fuzzy_duplicate_indices:
                    continue

                similarity = simhash_similarity(simhashes[i][1], simhashes[j][1])

                if similarity >= fuzzy_threshold:
                    # Mark as fuzzy duplicate
                    if preserve_first:
                        fuzzy_duplicate_indices.add(simhashes[j][0])
                    else:
                        fuzzy_duplicate_indices.add(simhashes[i][0])

                    # Track source
                    if track_sources:
                        dup_idx = simhashes[j][0] if preserve_first else simhashes[i][0]
                        dup_chunk = chunks[dup_idx]
                        source = dup_chunk.metadata.get('source', 'unknown')
                        source_type = dup_chunk.metadata.get('source_type', 'unknown')
                        key = f"{source_type}:{source} (fuzzy)"
                        stats.duplicates_by_source[key] = stats.duplicates_by_source.get(key, 0) + 1

        stats.fuzzy_duplicates_removed = len(fuzzy_duplicate_indices)

    # Build final deduplicated list
    all_duplicate_indices = exact_duplicate_indices | fuzzy_duplicate_indices
    deduplicated = [
        chunk for i, chunk in enumerate(chunks)
        if i not in all_duplicate_indices
    ]

    stats.total_output_chunks = len(deduplicated)

    return deduplicated, stats


def deduplicate_chunks_streaming(
    chunks_iterator,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 0.95,
    max_fuzzy_window: int = 1000,
) -> Tuple[List[Document], DeduplicationStats]:
    """
    Deduplicate chunks in a streaming fashion with bounded memory.

    For very large document sets, this function processes chunks
    incrementally using a sliding window for fuzzy matching.

    Args:
        chunks_iterator: Iterator yielding Document objects
        enable_fuzzy: If True, also remove near-duplicates
        fuzzy_threshold: Similarity threshold for fuzzy matching
        max_fuzzy_window: Maximum window size for fuzzy comparison

    Returns:
        Tuple of (deduplicated_chunks, statistics)
    """
    stats = DeduplicationStats()

    seen_hashes: Set[str] = set()
    fuzzy_window: List[Tuple[int, Document]] = []  # (simhash, chunk)
    deduplicated: List[Document] = []

    for chunk in chunks_iterator:
        stats.total_input_chunks += 1
        content_hash = compute_chunk_hash(chunk.page_content)

        # Check exact duplicate
        if content_hash in seen_hashes:
            stats.exact_duplicates_removed += 1
            continue

        seen_hashes.add(content_hash)

        # Check fuzzy duplicate if enabled
        is_fuzzy_dup = False
        if enable_fuzzy:
            simhash = compute_simhash(chunk.page_content)

            for window_simhash, _ in fuzzy_window:
                similarity = simhash_similarity(simhash, window_simhash)
                if similarity >= fuzzy_threshold:
                    is_fuzzy_dup = True
                    stats.fuzzy_duplicates_removed += 1
                    break

            if not is_fuzzy_dup:
                # Add to fuzzy window
                fuzzy_window.append((simhash, chunk))

                # Trim window if too large
                if len(fuzzy_window) > max_fuzzy_window:
                    fuzzy_window.pop(0)

        if not is_fuzzy_dup:
            deduplicated.append(chunk)

    stats.total_output_chunks = len(deduplicated)
    return deduplicated, stats


def find_duplicate_groups(
    chunks: List[Document],
    fuzzy_threshold: float = 0.95,
) -> Dict[str, List[Document]]:
    """
    Group chunks by content similarity.

    Useful for analysis and debugging - shows which chunks are
    duplicates of each other.

    Args:
        chunks: List of Document objects
        fuzzy_threshold: Similarity threshold for grouping

    Returns:
        Dictionary mapping content hash to list of duplicate chunks
    """
    groups: Dict[str, List[Document]] = defaultdict(list)

    for chunk in chunks:
        content_hash = compute_chunk_hash(chunk.page_content)
        groups[content_hash].append(chunk)

    # Filter to only return groups with duplicates
    return {k: v for k, v in groups.items() if len(v) > 1}


def log_deduplication_stats(
    stats: DeduplicationStats,
    verbose: bool = False,
    logger=None,
) -> None:
    """
    Log deduplication statistics.

    Args:
        stats: DeduplicationStats object
        verbose: If True, include per-source breakdown
        logger: Optional logger instance (uses print if None)
    """
    log = logger.info if logger else print

    log(f"\n{'='*60}")
    log("CHUNK DEDUPLICATION RESULTS")
    log(f"{'='*60}")
    log(f"Input chunks:  {stats.total_input_chunks}")
    log(f"Output chunks: {stats.total_output_chunks}")
    log(f"Exact duplicates removed: {stats.exact_duplicates_removed}")
    log(f"Fuzzy duplicates removed: {stats.fuzzy_duplicates_removed}")
    log(f"Total removed: {stats.total_removed} ({stats.deduplication_rate:.1f}%)")

    if verbose and stats.duplicates_by_source:
        log(f"\nDuplicates by source:")
        for source, count in sorted(
            stats.duplicates_by_source.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]:  # Top 20 sources
            log(f"  {source}: {count}")

        if len(stats.duplicates_by_source) > 20:
            remaining = len(stats.duplicates_by_source) - 20
            log(f"  ... and {remaining} more sources")


# Convenience function for simple deduplication
def simple_deduplicate(chunks: List[Document]) -> List[Document]:
    """
    Simple exact deduplication without statistics.

    Args:
        chunks: List of Document objects

    Returns:
        Deduplicated list of Document objects
    """
    deduplicated, _ = deduplicate_chunks(chunks, enable_fuzzy=False)
    return deduplicated


if __name__ == "__main__":
    # Simple test
    print("Chunk Deduplication Module")
    print("=" * 40)

    # Create test chunks
    test_chunks = [
        Document(page_content="Hello world, this is a test.", metadata={"source": "test1.md"}),
        Document(page_content="Hello world, this is a test.", metadata={"source": "test2.md"}),  # Exact dup
        Document(page_content="Hello world, this is a test!", metadata={"source": "test3.md"}),  # Near dup
        Document(page_content="Completely different content here.", metadata={"source": "test4.md"}),
    ]

    print(f"\nInput chunks: {len(test_chunks)}")

    # Test exact deduplication
    deduped, stats = deduplicate_chunks(test_chunks, enable_fuzzy=False)
    print(f"\nAfter exact deduplication: {len(deduped)}")
    print(stats)

    # Test fuzzy deduplication
    deduped_fuzzy, stats_fuzzy = deduplicate_chunks(test_chunks, enable_fuzzy=True, fuzzy_threshold=0.9)
    print(f"\nAfter fuzzy deduplication (threshold=0.9): {len(deduped_fuzzy)}")
    print(stats_fuzzy)
