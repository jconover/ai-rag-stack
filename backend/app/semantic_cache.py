"""Semantic Response Cache for LLM responses.

This module implements semantic caching for the RAG pipeline to reduce latency
and costs by caching LLM responses for semantically similar queries.

The cache works by:
1. Computing embeddings for incoming queries
2. Searching for similar previously cached queries using cosine similarity
3. Returning cached responses when similarity exceeds threshold (default 0.92)
4. Storing new query-response pairs with associated context hash

Cache key structure in Redis:
- sem_cache:index -> Set of all cached query hashes
- sem_cache:emb:{query_hash} -> Query embedding vector (JSON)
- sem_cache:meta:{query_hash} -> Metadata (context_hash, timestamp, model)
- sem_cache:resp:{query_hash} -> Cached response text

Performance characteristics:
- Cache hits: ~5-20ms (embedding lookup + similarity check)
- Cache misses: Full RAG pipeline time (typically 200-2000ms+)
- Memory: ~3KB per cached entry (768-dim embedding + response + metadata)

Usage:
    from app.semantic_cache import semantic_cache

    # Check cache before LLM call
    cached_response = semantic_cache.get(query, context_hash)
    if cached_response:
        return cached_response

    # After LLM call, store in cache
    semantic_cache.put(query, context_hash, response, model=model)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis

from app.config import settings
from app.redis_client import get_redis_client

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached response entry with metadata."""
    query_hash: str
    response: str
    context_hash: str
    model: str
    timestamp: float
    similarity_score: float = 0.0


@dataclass
class CacheStats:
    """Statistics for semantic cache performance."""
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    errors: int = 0
    avg_similarity_on_hit: float = 0.0
    avg_lookup_time_ms: float = 0.0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SemanticResponseCache:
    """Cache LLM responses based on semantic similarity of queries.

    This cache uses embeddings to find semantically similar previous queries
    and returns cached responses when similarity exceeds a configurable threshold.
    This provides significant latency reduction (80-95%) for similar queries.

    The cache stores:
    - Query embeddings for similarity computation
    - Context hash to ensure retrieved documents haven't changed
    - Original response text
    - Metadata (model, timestamp) for debugging and TTL management

    Attributes:
        SIMILARITY_THRESHOLD: Minimum cosine similarity for cache hit (default 0.92)
        TTL_SECONDS: Time-to-live for cache entries (default 3600s = 1 hour)
    """

    # Redis key prefixes for different cache components
    CACHE_PREFIX = "sem_cache:"
    INDEX_KEY = "sem_cache:index"
    EMB_PREFIX = "sem_cache:emb:"
    META_PREFIX = "sem_cache:meta:"
    RESP_PREFIX = "sem_cache:resp:"

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """Initialize semantic response cache.

        Args:
            similarity_threshold: Minimum cosine similarity for cache hit.
                                 Defaults to settings.semantic_cache_threshold (0.92)
            ttl_seconds: Time-to-live for cache entries in seconds.
                        Defaults to settings.semantic_cache_ttl (3600)
        """
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.semantic_cache_threshold
        )
        self.ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else settings.semantic_cache_ttl
        )

        self._enabled = settings.semantic_cache_enabled
        self._redis_client: Optional[redis.Redis] = None
        self._embeddings = None  # Lazy loaded

        # In-memory statistics
        self._stats = CacheStats()
        self._similarity_sum_on_hits = 0.0
        self._lookup_time_sum_ms = 0.0
        self._lookup_count = 0

        if self._enabled:
            self._init_redis()

        logger.info(
            f"SemanticResponseCache initialized: enabled={self._enabled}, "
            f"threshold={self.similarity_threshold}, ttl={self.ttl_seconds}s"
        )

    def _init_redis(self) -> None:
        """Initialize Redis connection using shared pool."""
        try:
            self._redis_client = get_redis_client()
            # Test connection
            self._redis_client.ping()
            logger.info("Semantic cache Redis connection initialized")
        except redis.ConnectionError as e:
            logger.warning(f"Failed to connect to Redis for semantic cache: {e}")
            self._redis_client = None
            self._stats.errors += 1
        except Exception as e:
            logger.warning(f"Error initializing semantic cache Redis: {e}")
            self._redis_client = None
            self._stats.errors += 1

    def _get_embeddings(self):
        """Lazy load shared embedding model."""
        if self._embeddings is None:
            try:
                from app.vectorstore import get_shared_embeddings
                self._embeddings = get_shared_embeddings()
            except Exception as e:
                logger.error(f"Failed to load embedding model for semantic cache: {e}")
                self._stats.errors += 1
                return None
        return self._embeddings

    def _compute_query_hash(self, query: str) -> str:
        """Compute a stable hash for a query string.

        Uses MD5 for fast hashing. The hash is used as a unique identifier
        for cached entries, not for security.

        Args:
            query: The query string

        Returns:
            16-character hex hash of the query
        """
        return hashlib.md5(query.encode('utf-8')).hexdigest()[:16]

    def _compute_embedding(self, query: str) -> Optional[List[float]]:
        """Compute embedding vector for a query.

        Args:
            query: The query string

        Returns:
            Embedding vector as list of floats, or None on error
        """
        embeddings = self._get_embeddings()
        if embeddings is None:
            return None

        try:
            # Use BGE instruction prefix for consistency with vectorstore
            if 'bge' in settings.embedding_model.lower():
                prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
            else:
                prefixed_query = query

            return embeddings.embed_query(prefixed_query)
        except Exception as e:
            logger.error(f"Error computing embedding for semantic cache: {e}")
            self._stats.errors += 1
            return None

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score in range [-1, 1]
        """
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    def _get_cached_embeddings(self) -> List[Tuple[str, List[float]]]:
        """Retrieve all cached embeddings from Redis.

        Returns:
            List of (query_hash, embedding_vector) tuples
        """
        if self._redis_client is None:
            return []

        try:
            # Get all query hashes from index
            index_data = self._redis_client.smembers(self.INDEX_KEY)
            if not index_data:
                return []

            results = []
            for hash_bytes in index_data:
                query_hash = hash_bytes.decode('utf-8') if isinstance(hash_bytes, bytes) else hash_bytes
                emb_key = f"{self.EMB_PREFIX}{query_hash}"
                emb_data = self._redis_client.get(emb_key)

                if emb_data:
                    try:
                        embedding = json.loads(
                            emb_data.decode('utf-8') if isinstance(emb_data, bytes) else emb_data
                        )
                        results.append((query_hash, embedding))
                    except json.JSONDecodeError:
                        logger.debug(f"Invalid embedding JSON for hash {query_hash}")
                        continue

            return results

        except redis.RedisError as e:
            logger.warning(f"Redis error getting cached embeddings: {e}")
            self._stats.errors += 1
            return []

    def _find_similar_query(
        self,
        query_embedding: List[float],
        context_hash: str
    ) -> Optional[Tuple[str, float]]:
        """Find the most similar cached query above threshold.

        Args:
            query_embedding: Embedding vector of the incoming query
            context_hash: Hash of the retrieval context

        Returns:
            Tuple of (query_hash, similarity_score) if found, None otherwise
        """
        cached_embeddings = self._get_cached_embeddings()
        if not cached_embeddings:
            return None

        best_match: Optional[Tuple[str, float]] = None
        best_score = 0.0

        for query_hash, cached_embedding in cached_embeddings:
            # Check if context hash matches (ensures same retrieved documents)
            meta_key = f"{self.META_PREFIX}{query_hash}"
            meta_data = self._redis_client.get(meta_key)

            if meta_data:
                try:
                    metadata = json.loads(
                        meta_data.decode('utf-8') if isinstance(meta_data, bytes) else meta_data
                    )
                    cached_context_hash = metadata.get('context_hash', '')

                    # Skip if context hash doesn't match
                    if cached_context_hash != context_hash:
                        continue

                except json.JSONDecodeError:
                    continue

            # Compute similarity
            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity > best_score and similarity >= self.similarity_threshold:
                best_score = similarity
                best_match = (query_hash, similarity)

        return best_match

    def get(
        self,
        query: str,
        context_hash: str
    ) -> Optional[str]:
        """Check if semantically similar query was already answered.

        Searches cached queries for one with:
        1. Cosine similarity above threshold (default 0.92)
        2. Same context hash (ensuring same retrieved documents)

        Args:
            query: The incoming query string
            context_hash: MD5 hash of the retrieval context

        Returns:
            Cached response string if found, None otherwise
        """
        if not self._enabled or self._redis_client is None:
            self._stats.misses += 1
            return None

        lookup_start = time.perf_counter()

        try:
            # Compute embedding for incoming query
            query_embedding = self._compute_embedding(query)
            if query_embedding is None:
                self._stats.misses += 1
                return None

            # Find similar cached query
            match = self._find_similar_query(query_embedding, context_hash)

            if match is None:
                self._stats.misses += 1
                lookup_time_ms = (time.perf_counter() - lookup_start) * 1000
                self._update_lookup_stats(lookup_time_ms)
                logger.debug(f"Semantic cache MISS for query: '{query[:50]}...'")
                return None

            query_hash, similarity = match

            # Retrieve cached response
            resp_key = f"{self.RESP_PREFIX}{query_hash}"
            cached_response = self._redis_client.get(resp_key)

            if cached_response is None:
                # Response expired but metadata remains - clean up
                self._cleanup_entry(query_hash)
                self._stats.misses += 1
                return None

            response_text = (
                cached_response.decode('utf-8')
                if isinstance(cached_response, bytes)
                else cached_response
            )

            # Update statistics
            self._stats.hits += 1
            self._similarity_sum_on_hits += similarity
            lookup_time_ms = (time.perf_counter() - lookup_start) * 1000
            self._update_lookup_stats(lookup_time_ms)

            logger.info(
                f"Semantic cache HIT: similarity={similarity:.4f}, "
                f"query='{query[:30]}...', time={lookup_time_ms:.1f}ms"
            )

            return response_text

        except redis.RedisError as e:
            logger.warning(f"Redis error on semantic cache get: {e}")
            self._stats.errors += 1
            self._stats.misses += 1
            return None
        except Exception as e:
            logger.warning(f"Unexpected error on semantic cache get: {e}")
            self._stats.errors += 1
            self._stats.misses += 1
            return None

    def put(
        self,
        query: str,
        context_hash: str,
        response: str,
        model: str = ""
    ) -> bool:
        """Store response with semantic indexing.

        Stores the query embedding, metadata, and response in Redis with TTL.

        Args:
            query: The query string
            context_hash: MD5 hash of the retrieval context
            response: The LLM-generated response to cache
            model: The model name used for generation (for debugging)

        Returns:
            True if successfully cached, False otherwise
        """
        if not self._enabled or self._redis_client is None:
            return False

        try:
            # Compute embedding for query
            query_embedding = self._compute_embedding(query)
            if query_embedding is None:
                return False

            # Generate query hash
            query_hash = self._compute_query_hash(query)

            # Prepare metadata
            metadata = {
                'context_hash': context_hash,
                'model': model,
                'timestamp': time.time(),
                'query_preview': query[:100],
            }

            # Store in Redis with pipeline for atomicity
            pipe = self._redis_client.pipeline()

            # Add to index
            pipe.sadd(self.INDEX_KEY, query_hash)

            # Store embedding
            emb_key = f"{self.EMB_PREFIX}{query_hash}"
            pipe.setex(emb_key, self.ttl_seconds, json.dumps(query_embedding))

            # Store metadata
            meta_key = f"{self.META_PREFIX}{query_hash}"
            pipe.setex(meta_key, self.ttl_seconds, json.dumps(metadata))

            # Store response
            resp_key = f"{self.RESP_PREFIX}{query_hash}"
            pipe.setex(resp_key, self.ttl_seconds, response.encode('utf-8'))

            pipe.execute()

            self._stats.stores += 1

            logger.info(
                f"Semantic cache stored: hash={query_hash}, "
                f"query='{query[:30]}...', ttl={self.ttl_seconds}s"
            )

            return True

        except redis.RedisError as e:
            logger.warning(f"Redis error on semantic cache put: {e}")
            self._stats.errors += 1
            return False
        except Exception as e:
            logger.warning(f"Unexpected error on semantic cache put: {e}")
            self._stats.errors += 1
            return False

    def _cleanup_entry(self, query_hash: str) -> None:
        """Remove a cache entry from all Redis keys.

        Args:
            query_hash: The query hash to remove
        """
        if self._redis_client is None:
            return

        try:
            pipe = self._redis_client.pipeline()
            pipe.srem(self.INDEX_KEY, query_hash)
            pipe.delete(f"{self.EMB_PREFIX}{query_hash}")
            pipe.delete(f"{self.META_PREFIX}{query_hash}")
            pipe.delete(f"{self.RESP_PREFIX}{query_hash}")
            pipe.execute()
            self._stats.evictions += 1
        except Exception as e:
            logger.debug(f"Error cleaning up cache entry {query_hash}: {e}")

    def _update_lookup_stats(self, lookup_time_ms: float) -> None:
        """Update running average of lookup times."""
        self._lookup_time_sum_ms += lookup_time_ms
        self._lookup_count += 1
        if self._lookup_count > 0:
            self._stats.avg_lookup_time_ms = self._lookup_time_sum_ms / self._lookup_count
        if self._stats.hits > 0:
            self._stats.avg_similarity_on_hit = self._similarity_sum_on_hits / self._stats.hits

    def clear(self) -> int:
        """Clear all semantic cache entries from Redis.

        Returns:
            Number of entries cleared
        """
        if self._redis_client is None:
            return 0

        try:
            # Get all hashes from index
            index_data = self._redis_client.smembers(self.INDEX_KEY)
            if not index_data:
                return 0

            count = len(index_data)

            # Delete all entries
            pipe = self._redis_client.pipeline()
            for hash_bytes in index_data:
                query_hash = hash_bytes.decode('utf-8') if isinstance(hash_bytes, bytes) else hash_bytes
                pipe.delete(f"{self.EMB_PREFIX}{query_hash}")
                pipe.delete(f"{self.META_PREFIX}{query_hash}")
                pipe.delete(f"{self.RESP_PREFIX}{query_hash}")

            # Clear index
            pipe.delete(self.INDEX_KEY)
            pipe.execute()

            # Reset statistics
            self._stats = CacheStats()
            self._similarity_sum_on_hits = 0.0
            self._lookup_time_sum_ms = 0.0
            self._lookup_count = 0

            logger.info(f"Cleared {count} entries from semantic cache")
            return count

        except redis.RedisError as e:
            logger.warning(f"Redis error clearing semantic cache: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        # Count current entries
        total_entries = 0
        if self._redis_client is not None:
            try:
                total_entries = self._redis_client.scard(self.INDEX_KEY) or 0
            except Exception:
                pass

        self._stats.total_entries = total_entries

        return {
            'enabled': self._enabled,
            'connected': self._redis_client is not None,
            'similarity_threshold': self.similarity_threshold,
            'ttl_seconds': self.ttl_seconds,
            'hits': self._stats.hits,
            'misses': self._stats.misses,
            'stores': self._stats.stores,
            'evictions': self._stats.evictions,
            'errors': self._stats.errors,
            'hit_rate': round(self._stats.hit_rate, 4),
            'avg_similarity_on_hit': round(self._stats.avg_similarity_on_hit, 4),
            'avg_lookup_time_ms': round(self._stats.avg_lookup_time_ms, 2),
            'total_entries': total_entries,
        }

    def is_connected(self) -> bool:
        """Check if Redis is connected and responsive."""
        if not self._enabled or self._redis_client is None:
            return False
        try:
            self._redis_client.ping()
            return True
        except Exception:
            return False


# Global singleton instance
semantic_cache = SemanticResponseCache()


def compute_context_hash(documents: List[Any]) -> str:
    """Compute a hash of the retrieval context for cache validation.

    This hash ensures that cached responses are only returned when
    the same documents would be retrieved. If documents change (new
    ingestion, different retrieval results), the cache is invalidated.

    Args:
        documents: List of Document objects from retrieval

    Returns:
        MD5 hash string of document contents
    """
    if not documents:
        return "empty"

    # Hash based on document content and sources
    content_parts = []
    for doc in documents:
        source = doc.metadata.get('source', '')
        # Use first 200 chars of content to avoid huge strings
        content_preview = doc.page_content[:200] if doc.page_content else ''
        content_parts.append(f"{source}:{content_preview}")

    combined = "|".join(content_parts)
    return hashlib.md5(combined.encode('utf-8')).hexdigest()
