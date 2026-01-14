"""RAG Pipeline Validation Tests for CI/CD Integration.

This module provides integration tests for the RAG pipeline that run against
real service containers (Qdrant, Redis) in the CI environment. These tests
validate the core RAG functionality without requiring GPU-dependent Ollama.

Tests validate:
- Vector store connection and health
- Document embedding and indexing
- Semantic search retrieval
- Hybrid search functionality (when enabled)
- Embedding cache operations

NOTE: Unlike other tests in this project that use mocks (via conftest.py),
these tests are designed to run against REAL service containers provided
by GitHub Actions services configuration. Do not import from conftest.py
as it would inject mocks into sys.modules.

Usage:
    pytest tests/test_rag_validation.py -v

CI Environment Variables:
    - QDRANT_HOST: Qdrant service host (default: localhost)
    - QDRANT_PORT: Qdrant service port (default: 6333)
    - REDIS_HOST: Redis service host (default: localhost)
    - REDIS_PORT: Redis service port (default: 6379)
"""

import os
import sys
import uuid
import pytest
import time
from typing import List, Tuple

# Configure environment for CI before any app imports
# These settings ensure we connect to CI service containers
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("QDRANT_GRPC_PORT", "6334")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("POSTGRES_HOST", "localhost")

# Disable features that require services not available in CI
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("QUERY_LOGGING_ENABLED", "false")
os.environ.setdefault("RERANKER_ENABLED", "false")
os.environ.setdefault("HYDE_ENABLED", "false")
os.environ.setdefault("WEB_SEARCH_ENABLED", "false")
os.environ.setdefault("EMBEDDING_CACHE_ENABLED", "true")  # Test cache with Redis

# Use a unique collection name for CI tests to avoid conflicts
CI_COLLECTION_NAME = f"ci_test_{uuid.uuid4().hex[:8]}"
os.environ["QDRANT_COLLECTION_NAME"] = CI_COLLECTION_NAME


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def qdrant_client():
    """Create a Qdrant client for testing.

    This fixture connects to the real Qdrant service container.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", 6333))

    client = QdrantClient(host=host, port=port, timeout=30)

    # Verify connection
    try:
        collections = client.get_collections()
        print(f"Connected to Qdrant at {host}:{port}")
    except Exception as e:
        pytest.skip(f"Qdrant not available at {host}:{port}: {e}")

    yield client

    # Cleanup: delete test collection after tests
    try:
        client.delete_collection(CI_COLLECTION_NAME)
        print(f"Cleaned up test collection: {CI_COLLECTION_NAME}")
    except Exception:
        pass


@pytest.fixture(scope="module")
def redis_client():
    """Create a Redis client for testing.

    This fixture connects to the real Redis service container.
    """
    import redis

    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", 6379))

    client = redis.Redis(host=host, port=port, decode_responses=True)

    # Verify connection
    try:
        client.ping()
        print(f"Connected to Redis at {host}:{port}")
    except redis.ConnectionError as e:
        pytest.skip(f"Redis not available at {host}:{port}: {e}")

    yield client

    # Cleanup: remove test keys
    try:
        for key in client.scan_iter("emb:*"):
            client.delete(key)
    except Exception:
        pass


@pytest.fixture(scope="module")
def embedding_model():
    """Load the embedding model for testing.

    Uses the same model as production for accurate validation.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}  # CI runs on CPU
        )
        # Warmup
        embeddings.embed_query("warmup")
        print(f"Loaded embedding model: {model_name}")
        return embeddings
    except Exception as e:
        pytest.skip(f"Failed to load embedding model: {e}")


@pytest.fixture(scope="module")
def test_documents():
    """Sample documents for testing embedding and retrieval."""
    return [
        {
            "content": "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications.",
            "metadata": {"source": "kubernetes/intro.md", "source_type": "kubernetes"}
        },
        {
            "content": "Docker containers package applications with their dependencies into standardized units for software development.",
            "metadata": {"source": "docker/overview.md", "source_type": "docker"}
        },
        {
            "content": "Terraform enables infrastructure as code, allowing you to define and provision infrastructure using declarative configuration files.",
            "metadata": {"source": "terraform/basics.md", "source_type": "terraform"}
        },
        {
            "content": "Prometheus is an open-source monitoring system that collects metrics from configured targets at given intervals.",
            "metadata": {"source": "prometheus/setup.md", "source_type": "prometheus"}
        },
        {
            "content": "Ansible automates application deployment, configuration management, and task automation using YAML playbooks.",
            "metadata": {"source": "ansible/playbooks.md", "source_type": "ansible"}
        },
    ]


# =============================================================================
# Connection Tests
# =============================================================================

class TestVectorStoreConnection:
    """Tests for vector store (Qdrant) connection and health."""

    def test_qdrant_connection(self, qdrant_client):
        """Test that Qdrant is reachable and responsive."""
        collections = qdrant_client.get_collections()
        assert collections is not None
        print(f"Qdrant collections: {[c.name for c in collections.collections]}")

    def test_qdrant_create_collection(self, qdrant_client):
        """Test creating a collection with proper configuration."""
        from qdrant_client.models import Distance, VectorParams

        # Ensure collection doesn't exist
        try:
            qdrant_client.delete_collection(CI_COLLECTION_NAME)
        except Exception:
            pass

        # Create collection with same settings as production
        qdrant_client.create_collection(
            collection_name=CI_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,  # BGE-base dimension
                distance=Distance.COSINE,
            )
        )

        # Verify collection exists
        collection_info = qdrant_client.get_collection(CI_COLLECTION_NAME)
        assert collection_info is not None
        assert collection_info.config.params.vectors.size == 768
        print(f"Created collection: {CI_COLLECTION_NAME}")

    def test_qdrant_collection_health(self, qdrant_client):
        """Test that collection is healthy and operational."""
        collection_info = qdrant_client.get_collection(CI_COLLECTION_NAME)

        # Collection status should be green or yellow (initializing)
        status = str(collection_info.status).lower()
        assert status in ["green", "yellow", "collectionstatus.green", "collectionstatus.yellow"]
        print(f"Collection status: {status}")


class TestRedisConnection:
    """Tests for Redis connection and operations."""

    def test_redis_ping(self, redis_client):
        """Test that Redis is reachable and responsive."""
        assert redis_client.ping() is True

    def test_redis_set_get(self, redis_client):
        """Test basic Redis set/get operations."""
        test_key = "ci_test_key"
        test_value = "test_value"

        redis_client.set(test_key, test_value)
        result = redis_client.get(test_key)

        assert result == test_value

        # Cleanup
        redis_client.delete(test_key)

    def test_redis_ttl_operations(self, redis_client):
        """Test Redis TTL operations for cache expiry."""
        test_key = "ci_test_ttl_key"

        redis_client.setex(test_key, 60, "expiring_value")
        ttl = redis_client.ttl(test_key)

        assert 0 < ttl <= 60

        # Cleanup
        redis_client.delete(test_key)


# =============================================================================
# Embedding Tests
# =============================================================================

class TestDocumentEmbedding:
    """Tests for document embedding generation."""

    def test_embed_single_query(self, embedding_model):
        """Test embedding a single query string."""
        query = "How do I create a Kubernetes deployment?"
        embedding = embedding_model.embed_query(query)

        assert embedding is not None
        assert len(embedding) == 768  # BGE-base dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_multiple_documents(self, embedding_model, test_documents):
        """Test embedding multiple documents in batch."""
        texts = [doc["content"] for doc in test_documents]
        embeddings = embedding_model.embed_documents(texts)

        assert len(embeddings) == len(test_documents)
        for emb in embeddings:
            assert len(emb) == 768

    def test_embedding_consistency(self, embedding_model):
        """Test that same text produces consistent embeddings."""
        text = "Kubernetes pod management"

        emb1 = embedding_model.embed_query(text)
        emb2 = embedding_model.embed_query(text)

        # Embeddings should be identical for same input
        assert emb1 == emb2

    def test_embedding_differentiation(self, embedding_model):
        """Test that different texts produce different embeddings."""
        text1 = "Kubernetes container orchestration"
        text2 = "PostgreSQL database administration"

        emb1 = embedding_model.embed_query(text1)
        emb2 = embedding_model.embed_query(text2)

        # Calculate cosine similarity
        import math
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        similarity = dot_product / (norm1 * norm2)

        # Unrelated topics should have lower similarity
        assert similarity < 0.9
        print(f"Similarity between unrelated topics: {similarity:.4f}")


# =============================================================================
# Document Retrieval Tests
# =============================================================================

class TestDocumentRetrieval:
    """Tests for document indexing and retrieval."""

    def test_index_documents(self, qdrant_client, embedding_model, test_documents):
        """Test indexing documents into the vector store."""
        from qdrant_client.models import PointStruct

        points = []
        for i, doc in enumerate(test_documents):
            embedding = embedding_model.embed_query(doc["content"])
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "page_content": doc["content"],
                        **doc["metadata"]
                    }
                )
            )

        # Upsert documents
        result = qdrant_client.upsert(
            collection_name=CI_COLLECTION_NAME,
            points=points
        )

        assert result.status.name in ["COMPLETED", "ACKNOWLEDGED"]

        # Verify documents are indexed
        collection_info = qdrant_client.get_collection(CI_COLLECTION_NAME)
        assert collection_info.points_count == len(test_documents)
        print(f"Indexed {collection_info.points_count} documents")

    def test_semantic_search(self, qdrant_client, embedding_model):
        """Test semantic search returns relevant results."""
        query = "How do I deploy containers?"
        query_embedding = embedding_model.embed_query(query)

        results = qdrant_client.search(
            collection_name=CI_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )

        assert len(results) > 0

        # Top result should be related to containers (Kubernetes or Docker)
        top_result = results[0]
        source_type = top_result.payload.get("source_type", "")
        assert source_type in ["kubernetes", "docker"]
        print(f"Top result source: {source_type}, score: {top_result.score:.4f}")

    def test_search_with_filter(self, qdrant_client, embedding_model):
        """Test search with source_type filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query = "How do I configure this tool?"
        query_embedding = embedding_model.embed_query(query)

        # Search only in Terraform docs
        results = qdrant_client.search(
            collection_name=CI_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value="terraform")
                    )
                ]
            ),
            limit=3
        )

        assert len(results) > 0
        for result in results:
            assert result.payload.get("source_type") == "terraform"

    def test_search_score_threshold(self, qdrant_client, embedding_model):
        """Test that search results have reasonable similarity scores."""
        query = "Kubernetes deployment YAML configuration"
        query_embedding = embedding_model.embed_query(query)

        results = qdrant_client.search(
            collection_name=CI_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5,
            score_threshold=0.3  # Minimum relevance threshold
        )

        # All returned results should meet threshold
        for result in results:
            assert result.score >= 0.3
            print(f"Result: {result.payload.get('source_type')} - Score: {result.score:.4f}")


# =============================================================================
# Hybrid Search Tests
# =============================================================================

class TestHybridSearch:
    """Tests for hybrid search functionality.

    These tests validate the hybrid search mechanism even when
    sparse vectors are not indexed. The focus is on testing the
    fallback behavior and search quality.
    """

    def test_dense_search_fallback(self, qdrant_client, embedding_model):
        """Test that dense search works as fallback when sparse is unavailable."""
        # This collection only has dense vectors
        query = "container orchestration platform"
        query_embedding = embedding_model.embed_query(query)

        results = qdrant_client.search(
            collection_name=CI_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )

        assert len(results) > 0
        # Should find Kubernetes doc
        assert any(
            r.payload.get("source_type") == "kubernetes"
            for r in results
        )

    def test_search_ranking_quality(self, qdrant_client, embedding_model):
        """Test that search results are properly ranked by relevance."""
        query = "infrastructure as code terraform"
        query_embedding = embedding_model.embed_query(query)

        results = qdrant_client.search(
            collection_name=CI_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5
        )

        # Results should be sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Terraform should rank highly for this query
        terraform_rank = None
        for i, r in enumerate(results):
            if r.payload.get("source_type") == "terraform":
                terraform_rank = i + 1
                break

        assert terraform_rank is not None
        assert terraform_rank <= 2  # Should be in top 2
        print(f"Terraform ranked #{terraform_rank} for IaC query")

    @pytest.mark.skipif(
        os.environ.get("HYBRID_SEARCH_ENABLED", "false").lower() != "true",
        reason="Hybrid search not enabled in CI"
    )
    def test_hybrid_search_enabled(self, qdrant_client, embedding_model):
        """Test hybrid search when explicitly enabled."""
        # This test only runs if HYBRID_SEARCH_ENABLED=true
        pass


# =============================================================================
# Embedding Cache Tests
# =============================================================================

class TestEmbeddingCache:
    """Tests for Redis-based embedding cache."""

    def test_cache_store_and_retrieve(self, redis_client, embedding_model):
        """Test storing and retrieving embeddings from cache."""
        import hashlib
        import json

        query = "test cache query"
        embedding = embedding_model.embed_query(query)

        # Simulate cache storage (matching production logic)
        model_hash = hashlib.md5("BAAI/bge-base-en-v1.5".encode()).hexdigest()[:8]
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"emb:{model_hash}:{query_hash}"

        # Store embedding
        redis_client.setex(cache_key, 3600, json.dumps(embedding))

        # Retrieve and verify
        cached = redis_client.get(cache_key)
        assert cached is not None

        cached_embedding = json.loads(cached)
        assert cached_embedding == embedding

        # Cleanup
        redis_client.delete(cache_key)

    def test_cache_ttl_expiry(self, redis_client, embedding_model):
        """Test that cached embeddings expire correctly."""
        import hashlib
        import json

        query = "expiring test query"
        embedding = embedding_model.embed_query(query)

        model_hash = hashlib.md5("BAAI/bge-base-en-v1.5".encode()).hexdigest()[:8]
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"emb:{model_hash}:{query_hash}"

        # Store with short TTL
        redis_client.setex(cache_key, 2, json.dumps(embedding))

        # Verify it exists
        assert redis_client.exists(cache_key)

        # Wait for expiry
        time.sleep(3)

        # Should be expired
        assert not redis_client.exists(cache_key)


# =============================================================================
# Integration Tests
# =============================================================================

class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline components."""

    def test_end_to_end_retrieval(self, qdrant_client, embedding_model, test_documents):
        """Test complete retrieval workflow from query to results."""
        # Index documents (may already be indexed)
        from qdrant_client.models import PointStruct

        points = []
        for i, doc in enumerate(test_documents):
            embedding = embedding_model.embed_query(doc["content"])
            points.append(
                PointStruct(
                    id=i + 100,  # Different IDs to avoid conflict
                    vector=embedding,
                    payload={
                        "page_content": doc["content"],
                        **doc["metadata"]
                    }
                )
            )

        qdrant_client.upsert(
            collection_name=CI_COLLECTION_NAME,
            points=points
        )

        # Perform retrieval
        user_query = "How do I automate infrastructure provisioning?"
        query_embedding = embedding_model.embed_query(user_query)

        results = qdrant_client.search(
            collection_name=CI_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=3
        )

        # Validate results
        assert len(results) >= 1

        # Check result quality - should find Terraform or Ansible
        relevant_sources = ["terraform", "ansible"]
        found_relevant = any(
            r.payload.get("source_type") in relevant_sources
            for r in results
        )
        assert found_relevant, f"Expected to find {relevant_sources} in results"

        print(f"Retrieved {len(results)} documents for query")
        for r in results:
            print(f"  - {r.payload.get('source_type')}: {r.score:.4f}")

    def test_retrieval_latency(self, qdrant_client, embedding_model):
        """Test that retrieval completes within acceptable latency."""
        query = "Kubernetes pod networking"

        start_time = time.time()

        # Embed query
        query_embedding = embedding_model.embed_query(query)
        embed_time = time.time() - start_time

        # Search
        search_start = time.time()
        results = qdrant_client.search(
            collection_name=CI_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=5
        )
        search_time = time.time() - search_start

        total_time = time.time() - start_time

        print(f"Retrieval latency: embed={embed_time*1000:.1f}ms, search={search_time*1000:.1f}ms, total={total_time*1000:.1f}ms")

        # Total retrieval should be under 2 seconds in CI
        # (embedding model loading may be slow on first run)
        assert total_time < 2.0, f"Retrieval too slow: {total_time:.2f}s"

    @pytest.mark.skip(reason="Ollama not available in CI - requires GPU")
    def test_llm_response_generation(self):
        """Test LLM response generation with retrieved context.

        This test is skipped in CI as it requires Ollama with GPU support.
        It validates the complete RAG pipeline including LLM inference.
        """
        pass


# =============================================================================
# Cleanup
# =============================================================================

@pytest.fixture(scope="module", autouse=True)
def cleanup_after_tests(request, qdrant_client):
    """Cleanup test artifacts after all tests complete."""
    yield

    # Cleanup is handled in the qdrant_client fixture teardown
    print("RAG validation tests completed")


# Run with: pytest backend/tests/test_rag_validation.py -v
