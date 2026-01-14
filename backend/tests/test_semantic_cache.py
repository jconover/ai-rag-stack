"""Test suite for Semantic Response Cache.

Tests the SemanticResponseCache class which provides semantic caching
for LLM responses based on query similarity.

All external dependencies (Redis, embedding model) are mocked to ensure
tests run quickly and reliably without requiring live services.

This test module does NOT use conftest mocking for app.semantic_cache
since we want to test the actual implementation. Instead, we mock
the dependencies (Redis, embeddings) directly.
"""

import hashlib
import json
import os
import sys

# Set up environment before imports
os.environ.setdefault("SEMANTIC_CACHE_ENABLED", "true")
os.environ.setdefault("SEMANTIC_CACHE_THRESHOLD", "0.92")
os.environ.setdefault("SEMANTIC_CACHE_TTL", "3600")

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

import numpy as np


# Define mock settings class for use in tests
class MockSettings:
    """Mock settings for semantic cache tests."""
    semantic_cache_enabled = True
    semantic_cache_threshold = 0.92
    semantic_cache_ttl = 3600
    embedding_model = "BAAI/bge-base-en-v1.5"


@pytest.fixture
def mock_redis():
    """Create a mock Redis client with in-memory storage."""
    storage = {}
    sets_storage = {}

    def mock_get(key):
        return storage.get(key)

    def mock_setex(key, ttl, value):
        storage[key] = value
        return True

    def mock_delete(*keys):
        count = 0
        for key in keys:
            if key in storage:
                del storage[key]
                count += 1
        return count

    def mock_sadd(key, *values):
        if key not in sets_storage:
            sets_storage[key] = set()
        for v in values:
            sets_storage[key].add(v)
        return len(values)

    def mock_smembers(key):
        return sets_storage.get(key, set())

    def mock_srem(key, *values):
        if key in sets_storage:
            for v in values:
                sets_storage[key].discard(v)
        return len(values)

    def mock_scard(key):
        return len(sets_storage.get(key, set()))

    def mock_pipeline():
        pipe = MagicMock()
        commands = []

        def record_command(cmd_name, *args):
            commands.append((cmd_name, args))
            return pipe

        pipe.sadd = lambda *args: record_command('sadd', *args)
        pipe.setex = lambda *args: record_command('setex', *args)
        pipe.delete = lambda *args: record_command('delete', *args)
        pipe.srem = lambda *args: record_command('srem', *args)

        def execute():
            for cmd_name, args in commands:
                if cmd_name == 'sadd':
                    mock_sadd(*args)
                elif cmd_name == 'setex':
                    mock_setex(*args)
                elif cmd_name == 'delete':
                    mock_delete(*args)
                elif cmd_name == 'srem':
                    mock_srem(*args)
            commands.clear()
            return [True] * len(commands)

        pipe.execute = execute
        return pipe

    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.get = mock_get
    redis_mock.setex = mock_setex
    redis_mock.delete = mock_delete
    redis_mock.sadd = mock_sadd
    redis_mock.smembers = mock_smembers
    redis_mock.srem = mock_srem
    redis_mock.scard = mock_scard
    redis_mock.pipeline = mock_pipeline

    # Expose storage for test inspection
    redis_mock._storage = storage
    redis_mock._sets_storage = sets_storage

    return redis_mock


@pytest.fixture
def mock_embeddings():
    """Create a mock embedding model."""
    embeddings = MagicMock()

    def create_embedding(text):
        # Create deterministic embedding based on text hash
        hash_val = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to floats for embedding
        np.random.seed(int(hash_val[:8], 16))
        return np.random.randn(768).tolist()

    embeddings.embed_query = create_embedding
    return embeddings


class TestSemanticCacheInitialization:
    """Tests for SemanticResponseCache initialization."""

    def test_cache_initialization_when_enabled(self, mock_redis, mock_embeddings):
        """Test cache initializes correctly when enabled."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                assert cache._enabled is True
                assert cache.similarity_threshold == 0.92
                assert cache.ttl_seconds == 3600

    def test_cache_initialization_when_disabled(self, mock_redis):
        """Test cache handles disabled state correctly."""
        mock_settings = MockSettings()
        mock_settings.semantic_cache_enabled = False

        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache
                cache = SemanticResponseCache()
                assert cache._enabled is False

    def test_cache_custom_threshold(self, mock_redis):
        """Test cache accepts custom similarity threshold."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache(similarity_threshold=0.85)
                assert cache.similarity_threshold == 0.85

    def test_cache_custom_ttl(self, mock_redis):
        """Test cache accepts custom TTL."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache(ttl_seconds=7200)
                assert cache.ttl_seconds == 7200


class TestCachePutOperation:
    """Tests for storing responses in the cache."""

    def test_put_stores_response(self, mock_redis, mock_embeddings):
        """Test that put() stores response correctly."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                cache._embeddings = mock_embeddings

                result = cache.put(
                    query="What is Kubernetes?",
                    context_hash="abc123",
                    response="Kubernetes is a container orchestration platform.",
                    model="llama3.1:8b"
                )

                assert result is True
                assert cache._stats.stores == 1

    def test_put_increments_store_count(self, mock_redis, mock_embeddings):
        """Test that put() increments store count."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                cache._embeddings = mock_embeddings

                cache.put("Query 1", "hash1", "Response 1")
                cache.put("Query 2", "hash2", "Response 2")

                assert cache._stats.stores == 2

    def test_put_returns_false_when_disabled(self, mock_redis):
        """Test that put() returns False when cache is disabled."""
        mock_settings = MockSettings()
        mock_settings.semantic_cache_enabled = False

        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache
                cache = SemanticResponseCache()
                result = cache.put("query", "hash", "response")
                assert result is False


class TestCacheGetOperation:
    """Tests for retrieving responses from the cache."""

    def test_get_returns_none_on_miss(self, mock_redis, mock_embeddings):
        """Test that get() returns None when no match found."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                cache._embeddings = mock_embeddings

                result = cache.get("What is Kubernetes?", "abc123")

                assert result is None
                assert cache._stats.misses == 1

    def test_get_returns_cached_response_on_hit(self, mock_redis, mock_embeddings):
        """Test that get() returns cached response on semantic match."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache(similarity_threshold=0.5)  # Low threshold for testing
                cache._embeddings = mock_embeddings

                # Store a response
                cache.put(
                    query="What is Kubernetes?",
                    context_hash="abc123",
                    response="Kubernetes is awesome!",
                    model="llama3.1:8b"
                )

                # Retrieve with exact same query (should hit)
                result = cache.get("What is Kubernetes?", "abc123")

                # With deterministic embeddings, exact match should hit
                assert result == "Kubernetes is awesome!"
                assert cache._stats.hits == 1

    def test_get_returns_none_when_disabled(self, mock_redis):
        """Test that get() returns None when cache is disabled."""
        mock_settings = MockSettings()
        mock_settings.semantic_cache_enabled = False

        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache
                cache = SemanticResponseCache()
                result = cache.get("query", "hash")
                assert result is None
                assert cache._stats.misses == 1

    def test_get_respects_context_hash(self, mock_redis, mock_embeddings):
        """Test that get() only returns cached response if context hash matches."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache(similarity_threshold=0.5)
                cache._embeddings = mock_embeddings

                # Store with one context hash
                cache.put(
                    query="What is Kubernetes?",
                    context_hash="original_context",
                    response="Kubernetes is awesome!",
                    model="llama3.1:8b"
                )

                # Try to retrieve with different context hash
                result = cache.get("What is Kubernetes?", "different_context")

                # Should miss because context hash doesn't match
                assert result is None


class TestCacheStatistics:
    """Tests for cache statistics tracking."""

    def test_stats_tracks_hits_and_misses(self, mock_redis, mock_embeddings):
        """Test that statistics correctly track hits and misses."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache(similarity_threshold=0.5)
                cache._embeddings = mock_embeddings

                # Miss
                cache.get("Query 1", "hash1")
                assert cache._stats.misses == 1

                # Store
                cache.put("Query 2", "hash2", "Response 2")
                assert cache._stats.stores == 1

                # Hit (same query)
                cache.get("Query 2", "hash2")
                assert cache._stats.hits == 1

    def test_get_stats_returns_complete_stats(self, mock_redis, mock_embeddings):
        """Test that get_stats() returns complete statistics."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                stats = cache.get_stats()

                assert 'enabled' in stats
                assert 'connected' in stats
                assert 'similarity_threshold' in stats
                assert 'ttl_seconds' in stats
                assert 'hits' in stats
                assert 'misses' in stats
                assert 'stores' in stats
                assert 'hit_rate' in stats

    def test_hit_rate_calculation(self, mock_redis, mock_embeddings):
        """Test that hit rate is calculated correctly."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache(similarity_threshold=0.5)
                cache._embeddings = mock_embeddings

                # 2 misses
                cache.get("Q1", "h1")
                cache.get("Q2", "h2")

                # Store and hit
                cache.put("Q3", "h3", "R3")
                cache.get("Q3", "h3")

                # 1 hit out of 3 lookups = 33.33%
                stats = cache.get_stats()
                assert stats['hits'] == 1
                assert stats['misses'] == 2
                # hit_rate = 1 / (1 + 2) = 0.3333
                assert 0.30 <= stats['hit_rate'] <= 0.35


class TestCacheClearOperation:
    """Tests for clearing the cache."""

    def test_clear_removes_all_entries(self, mock_redis, mock_embeddings):
        """Test that clear() removes all cached entries."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                cache._embeddings = mock_embeddings

                # Store some entries
                cache.put("Q1", "h1", "R1")
                cache.put("Q2", "h2", "R2")
                cache.put("Q3", "h3", "R3")

                # Clear
                count = cache.clear()

                # Should have cleared 3 entries
                assert count == 3
                assert cache._stats.hits == 0
                assert cache._stats.misses == 0
                assert cache._stats.stores == 0

    def test_clear_resets_statistics(self, mock_redis, mock_embeddings):
        """Test that clear() resets statistics."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                cache._embeddings = mock_embeddings

                cache.put("Q1", "h1", "R1")
                cache.get("Q1", "h1")
                cache.get("Q2", "h2")

                cache.clear()

                stats = cache.get_stats()
                assert stats['hits'] == 0
                assert stats['misses'] == 0
                assert stats['stores'] == 0


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        from app.semantic_cache import SemanticResponseCache

        cache = SemanticResponseCache.__new__(SemanticResponseCache)
        vec = [1.0, 2.0, 3.0]

        similarity = cache._cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        from app.semantic_cache import SemanticResponseCache

        cache = SemanticResponseCache.__new__(SemanticResponseCache)
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]

        similarity = cache._cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.0001

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1.0."""
        from app.semantic_cache import SemanticResponseCache

        cache = SemanticResponseCache.__new__(SemanticResponseCache)
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]

        similarity = cache._cosine_similarity(vec1, vec2)
        assert abs(similarity + 1.0) < 0.0001

    def test_cosine_similarity_handles_zero_vector(self):
        """Test cosine similarity handles zero vectors gracefully."""
        from app.semantic_cache import SemanticResponseCache

        cache = SemanticResponseCache.__new__(SemanticResponseCache)
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]

        similarity = cache._cosine_similarity(vec1, vec2)
        assert similarity == 0.0


class TestComputeContextHash:
    """Tests for context hash computation."""

    def test_compute_context_hash_empty_documents(self):
        """Test context hash for empty document list."""
        from app.semantic_cache import compute_context_hash

        result = compute_context_hash([])
        assert result == "empty"

    def test_compute_context_hash_single_document(self):
        """Test context hash for single document."""
        from app.semantic_cache import compute_context_hash

        doc = Mock()
        doc.page_content = "Test content"
        doc.metadata = {"source": "/test/file.md"}

        result = compute_context_hash([doc])
        assert len(result) == 32  # MD5 hex length

    def test_compute_context_hash_deterministic(self):
        """Test that context hash is deterministic."""
        from app.semantic_cache import compute_context_hash

        doc1 = Mock()
        doc1.page_content = "Test content"
        doc1.metadata = {"source": "/test/file.md"}

        doc2 = Mock()
        doc2.page_content = "Test content"
        doc2.metadata = {"source": "/test/file.md"}

        hash1 = compute_context_hash([doc1])
        hash2 = compute_context_hash([doc2])

        assert hash1 == hash2

    def test_compute_context_hash_different_for_different_content(self):
        """Test that context hash differs for different content."""
        from app.semantic_cache import compute_context_hash

        doc1 = Mock()
        doc1.page_content = "Content A"
        doc1.metadata = {"source": "/test/file.md"}

        doc2 = Mock()
        doc2.page_content = "Content B"
        doc2.metadata = {"source": "/test/file.md"}

        hash1 = compute_context_hash([doc1])
        hash2 = compute_context_hash([doc2])

        assert hash1 != hash2


class TestConnectionHandling:
    """Tests for Redis connection handling."""

    def test_is_connected_when_redis_available(self, mock_redis, mock_embeddings):
        """Test is_connected returns True when Redis is available."""
        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                assert cache.is_connected() is True

    def test_is_connected_when_disabled(self, mock_redis):
        """Test is_connected returns False when cache is disabled."""
        mock_settings = MockSettings()
        mock_settings.semantic_cache_enabled = False

        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_redis):
                from app.semantic_cache import SemanticResponseCache
                cache = SemanticResponseCache()
                assert cache.is_connected() is False

    def test_handles_redis_connection_error(self):
        """Test cache handles Redis connection errors gracefully."""
        import redis

        def raise_connection_error():
            raise redis.ConnectionError("Connection refused")

        mock_client = MagicMock()
        mock_client.ping.side_effect = raise_connection_error

        mock_settings = MockSettings()
        with patch('app.semantic_cache.settings', mock_settings):
            with patch('app.semantic_cache.get_redis_client', return_value=mock_client):
                from app.semantic_cache import SemanticResponseCache

                cache = SemanticResponseCache()
                # Should handle error gracefully
                assert cache._redis_client is None
                assert cache._stats.errors == 1


class TestCacheQueryHash:
    """Tests for query hash computation."""

    def test_compute_query_hash_deterministic(self):
        """Test query hash is deterministic."""
        from app.semantic_cache import SemanticResponseCache

        cache = SemanticResponseCache.__new__(SemanticResponseCache)

        hash1 = cache._compute_query_hash("What is Kubernetes?")
        hash2 = cache._compute_query_hash("What is Kubernetes?")

        assert hash1 == hash2

    def test_compute_query_hash_different_for_different_queries(self):
        """Test query hash differs for different queries."""
        from app.semantic_cache import SemanticResponseCache

        cache = SemanticResponseCache.__new__(SemanticResponseCache)

        hash1 = cache._compute_query_hash("What is Kubernetes?")
        hash2 = cache._compute_query_hash("What is Docker?")

        assert hash1 != hash2

    def test_compute_query_hash_length(self):
        """Test query hash has expected length."""
        from app.semantic_cache import SemanticResponseCache

        cache = SemanticResponseCache.__new__(SemanticResponseCache)

        query_hash = cache._compute_query_hash("Test query")
        assert len(query_hash) == 16  # First 16 chars of MD5


# Run with: pytest backend/tests/test_semantic_cache.py -v
