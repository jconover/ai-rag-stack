"""Pytest configuration and shared fixtures for API tests.

This module provides shared fixtures for testing the DevOps AI Assistant API,
including mocks for external dependencies (Ollama, Qdrant, Redis, PostgreSQL).

IMPORTANT: Module-level patches are applied BEFORE pytest collects tests
to prevent connection attempts during import of app modules.
"""
import sys
import os

# Add backend app to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# CRITICAL: Set environment variables BEFORE any imports
# =============================================================================

# Set environment variables to disable features that require external connections
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("QUERY_LOGGING_ENABLED", "false")
os.environ.setdefault("EMBEDDING_CACHE_ENABLED", "false")
os.environ.setdefault("RERANKER_ENABLED", "false")
os.environ.setdefault("HYBRID_SEARCH_ENABLED", "false")
os.environ.setdefault("HYDE_ENABLED", "false")
os.environ.setdefault("WEB_SEARCH_ENABLED", "false")
os.environ.setdefault("ENABLE_RETRIEVAL_METRICS", "false")

# =============================================================================
# CRITICAL: Inject mock modules into sys.modules BEFORE any app imports
# This MUST happen at module level, before pytest collection
# =============================================================================

from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass, field
from typing import List, Any, Optional


def _create_mock_vector_store():
    """Create a comprehensive mock VectorStore instance."""
    mock_store = MagicMock()
    mock_store.is_connected.return_value = True
    mock_store.get_stats.return_value = {
        "collection_name": "test_collection",
        "vectors_count": 1000,
        "points_count": 1000,
    }

    # Mock search methods to return empty results by default
    def mock_search_with_cache_info(query, top_k=5, min_score=0.3, source_type=None, source=None):
        return [], False

    def mock_hybrid_search_with_cache_info(query, top_k=5, min_score=0.3, source_type=None, source=None, alpha=0.5):
        return [], False

    mock_store.search_with_cache_info = MagicMock(side_effect=mock_search_with_cache_info)
    mock_store.hybrid_search_with_cache_info = MagicMock(side_effect=mock_hybrid_search_with_cache_info)
    mock_store.search = MagicMock(return_value=[])
    mock_store.search_with_scores = MagicMock(return_value=[])
    mock_store.hybrid_search = MagicMock(return_value=[])

    return mock_store


# Create module-level mock instances (singletons)
mock_vector_store_instance = _create_mock_vector_store()

# Create mock vectorstore module - this is a dependency of app.rag
mock_vectorstore_module = MagicMock()
mock_vectorstore_module.vector_store = mock_vector_store_instance
mock_vectorstore_module.VectorStore = MagicMock(return_value=mock_vector_store_instance)
mock_vectorstore_module.RedisEmbeddingCache = MagicMock()
mock_vectorstore_module.EmbeddingResult = MagicMock()

# Create mock reranker module - this is a dependency of app.rag
mock_reranker_module = MagicMock()
mock_reranker_module.rerank_documents = MagicMock(return_value=[])
mock_reranker_module.get_reranker = MagicMock(return_value=None)
mock_reranker_module.Reranker = MagicMock()

# Create mock query_expansion module - this is a dependency of app.rag
mock_query_expansion_module = MagicMock()
mock_hyde_expander = MagicMock()
mock_hyde_expander.expand_sync.return_value = MagicMock(
    expanded=False,
    hypothetical_document=None,
    generation_time_ms=0.0,
    skip_reason="disabled",
    error=None
)
mock_query_expansion_module.hyde_expander = mock_hyde_expander
mock_query_expansion_module.HyDEExpander = MagicMock()

# Create mock web_search module - this is a dependency of app.rag
mock_web_search_module = MagicMock()
mock_web_searcher = MagicMock()
mock_web_searcher.should_search.return_value = (False, None)
mock_web_searcher.search_sync.return_value = MagicMock(
    triggered=False,
    results=[],
    search_time_ms=0.0,
    error=None
)
mock_web_search_module.web_searcher = mock_web_searcher
mock_web_search_module.WebSearcher = MagicMock()

# Create mock metrics module - this is a dependency of app.rag
mock_metrics_module = MagicMock()
mock_metrics_module.retrieval_metrics_logger = MagicMock()
mock_metrics_module.RetrievalTimer = MagicMock()
mock_metrics_module.RetrievalMetrics = MagicMock()


def _create_mock_rag_pipeline():
    """Create a mock RAGPipeline instance."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_ollama_connected.return_value = True
    mock_pipeline.list_models.return_value = [
        {"name": "llama3.1:8b", "size": "4.7GB", "modified_at": "2024-01-01"},
        {"name": "mistral:7b", "size": "4.1GB", "modified_at": "2024-01-01"},
    ]
    mock_pipeline.get_reranker_status.return_value = {
        "enabled": True,
        "loaded": True,
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }

    # Mock async generate_response
    async def mock_generate_response(*args, **kwargs):
        return {
            "response": "This is a test response about Kubernetes deployments.",
            "model": "llama3.1:8b",
            "context_used": True,
            "sources": [
                {
                    "source": "kubernetes/docs/deployments.md",
                    "source_type": "kubernetes",
                    "content_preview": "A Deployment provides declarative updates...",
                    "similarity_score": 0.89
                }
            ],
            "retrieval_metrics": {
                "initial_candidates": 20,
                "after_reranking": 5,
                "reranker_used": True,
                "avg_similarity_score": 0.85,
                "retrieval_time_ms": 45.2,
                "rerank_time_ms": 12.3
            }
        }

    mock_pipeline.generate_response = AsyncMock(side_effect=mock_generate_response)

    # Mock async generate_response_stream
    async def mock_generate_response_stream(*args, **kwargs):
        yield {"response": "Test ", "done": False}
        yield {"response": "response.", "done": True}

    mock_pipeline.generate_response_stream = MagicMock(return_value=mock_generate_response_stream())

    return mock_pipeline


# Create mock rag_pipeline instance
mock_rag_pipeline_instance = _create_mock_rag_pipeline()

# Create mock rag module
mock_rag_module = MagicMock()
mock_rag_module.rag_pipeline = mock_rag_pipeline_instance
mock_rag_module.RAGPipeline = MagicMock(return_value=mock_rag_pipeline_instance)
mock_rag_module.RetrievalResult = MagicMock()

# Create mock database module
mock_database_module = MagicMock()
mock_database_module.init_db = AsyncMock(return_value=None)
mock_database_module.close_db = AsyncMock(return_value=None)
mock_database_module.get_db = MagicMock()
mock_database_module.get_db_context = MagicMock()
mock_database_module.check_postgres_connection = AsyncMock(return_value=True)  # Async function
mock_database_module.get_postgres_pool_stats = AsyncMock(return_value={  # Also async
    "host": "localhost",
    "port": 5432,
    "database": "devops_ai",
    "pool_size": 5,
    "max_overflow": 10,
    "checked_in": 5,
    "checked_out": 0,
    "overflow": 0,
    "pool_timeout": 30
})

# Create mock db_models module - SQLAlchemy models
mock_db_models_module = MagicMock()
mock_db_models_module.QueryLog = MagicMock()
mock_db_models_module.Feedback = MagicMock()
mock_db_models_module.Experiment = MagicMock()
mock_db_models_module.ExperimentAssignment = MagicMock()
mock_db_models_module.ExperimentResult = MagicMock()
mock_db_models_module.ExperimentStatus = MagicMock()
mock_db_models_module.ExperimentType = MagicMock()
mock_db_models_module.User = MagicMock()
mock_db_models_module.UserSession = MagicMock()
mock_db_models_module.APIKey = MagicMock()

# Create mock auth module
mock_auth_module = MagicMock()
mock_auth_module.auth_service = MagicMock()
mock_auth_module.hash_password = MagicMock(return_value="hashed")
mock_auth_module.verify_password = MagicMock(return_value=True)
mock_auth_module.generate_api_key = MagicMock(return_value="test_api_key")
mock_auth_module.hash_token = MagicMock(return_value="hashed_token")

# These are FastAPI dependencies - they need to be async functions that return appropriate values
async def mock_get_current_user():
    return MagicMock(id="test-user-id", email="test@example.com")

async def mock_get_optional_user():
    return None  # Optional user, return None for unauthenticated

mock_auth_module.get_current_user = mock_get_current_user
mock_auth_module.get_optional_user = mock_get_optional_user

# Create mock feedback module
mock_feedback_module = MagicMock()
mock_feedback_module.feedback_log = MagicMock()
mock_feedback_module.get_feedback_summary = MagicMock(return_value={})

# Create mock redis_client module - shared Redis connection pools
mock_redis_pool = MagicMock()
mock_redis_pool.max_connections = 10
mock_redis_pool._in_use_connections = set()
mock_redis_pool._available_connections = []

mock_redis_client_obj = MagicMock()
mock_redis_client_obj.ping.return_value = True
mock_redis_client_obj.lrange.return_value = []
mock_redis_client_obj.pipeline.return_value = MagicMock()
mock_redis_client_obj.get.return_value = None
mock_redis_client_obj.set.return_value = True
mock_redis_client_obj.delete.return_value = 1

mock_redis_client_module = MagicMock()
mock_redis_client_module.get_redis_pool = MagicMock(return_value=mock_redis_pool)
mock_redis_client_module.get_redis_string_pool = MagicMock(return_value=mock_redis_pool)
mock_redis_client_module.get_redis_client = MagicMock(return_value=mock_redis_client_obj)
mock_redis_client_module.get_redis_string_client = MagicMock(return_value=mock_redis_client_obj)
mock_redis_client_module.is_redis_connected = MagicMock(return_value=True)
mock_redis_client_module.close_redis_pool = MagicMock()
mock_redis_client_module.close_async_redis_pool = AsyncMock()
mock_redis_client_module.get_redis_pool_stats = MagicMock(return_value={
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "max_connections": 10,
    "current_connections": 0,
    "available_connections": 5,
    "bytes_pool": {"current_connections": 0, "available_connections": 2},
    "string_pool": {"current_connections": 0, "available_connections": 3},
    "async_pool": {"status": "not_initialized"},
})

# Create mock drift_detection module
mock_drift_status = MagicMock()
mock_drift_status.STABLE = "stable"
mock_drift_status.WARNING = "warning"
mock_drift_status.DRIFT_DETECTED = "drift_detected"
mock_drift_status.INSUFFICIENT_DATA = "insufficient_data"
mock_drift_status.NO_BASELINE = "no_baseline"

mock_drift_detector = MagicMock()
mock_drift_detector.record_score = MagicMock()
mock_drift_detector.check_drift = AsyncMock(return_value={
    "status": "stable",
    "message": "Distribution stable",
    "checked_at": "2026-01-12T00:00:00Z",
})
mock_drift_detector.set_baseline = AsyncMock(return_value=True)
mock_drift_detector.get_status = MagicMock(return_value={
    "scores_recorded": 0,
    "has_baseline": False,
})
mock_drift_detector.get_history = AsyncMock(return_value=[])
mock_drift_detector.reset = AsyncMock()

mock_drift_detection_module = MagicMock()
mock_drift_detection_module.drift_detector = mock_drift_detector
mock_drift_detection_module.DriftStatus = mock_drift_status
mock_drift_detection_module.DriftDetector = MagicMock()
mock_drift_detection_module.DriftMetrics = MagicMock()

# Create mock circuit_breaker module
# Need to return plain dict values, not MagicMock, for Pydantic validation
mock_circuit_breaker_module = MagicMock()

# Create proper mock status dict matching CircuitBreaker.get_status() return format
def _make_mock_breaker_status(name: str) -> dict:
    return {
        "name": name,
        "state": "closed",
        "config": {
            "failure_threshold": 5,
            "success_threshold": 2,
            "reset_timeout_seconds": 30.0,
            "retry_attempts": 3,
        },
        "stats": {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "success_rate": None,
        },
        "time_until_retry": None,
    }

mock_circuit_breaker_module.get_circuit_breaker_states = MagicMock(return_value={
    "ollama": _make_mock_breaker_status("ollama"),
    "qdrant": _make_mock_breaker_status("qdrant"),
    "tavily": _make_mock_breaker_status("tavily"),
})
mock_circuit_breaker_module.get_circuit_breakers_healthy = MagicMock(return_value=True)
mock_circuit_breaker_module.reset_all_circuit_breakers = MagicMock()
mock_circuit_breaker_module.CircuitBreaker = MagicMock()
mock_circuit_breaker_module.CircuitBreakerOpen = MagicMock()

# Create mock analytics module
mock_metrics_collector = MagicMock()
mock_metrics_collector.record_request = MagicMock()
mock_metrics_collector.record_error = MagicMock()
mock_metrics_collector.get_realtime_analytics = MagicMock(return_value={
    "requests_per_minute": 0,
    "avg_latency_ms": 0,
    "error_rate": 0,
})

mock_analytics_module = MagicMock()
mock_analytics_module.get_metrics_collector = MagicMock(return_value=mock_metrics_collector)
mock_analytics_module.MetricsCollector = MagicMock(return_value=mock_metrics_collector)

# Create mock device_utils module
mock_device_utils_module = MagicMock()
mock_device_utils_module.get_optimal_device = MagicMock(return_value="cpu")
mock_device_utils_module.get_device_info = MagicMock(return_value={
    "pytorch_available": True,
    "cuda_available": False,
    "mps_available": False,
    "recommended_device": "cpu",
    "current_embedding_device": "cpu",
    "current_reranker_device": "cpu",
})
mock_device_utils_module.log_device_configuration = MagicMock()
mock_device_utils_module.get_actual_embedding_device = MagicMock(return_value="cpu")
mock_device_utils_module.get_actual_reranker_device = MagicMock(return_value="cpu")

# Inject mocked modules into sys.modules BEFORE any app imports
# This prevents the real modules from being imported and connecting to services
sys.modules['app.vectorstore'] = mock_vectorstore_module
sys.modules['app.reranker'] = mock_reranker_module
sys.modules['app.query_expansion'] = mock_query_expansion_module
sys.modules['app.web_search'] = mock_web_search_module
sys.modules['app.metrics'] = mock_metrics_module
sys.modules['app.rag'] = mock_rag_module
sys.modules['app.database'] = mock_database_module
sys.modules['app.db_models'] = mock_db_models_module
sys.modules['app.auth'] = mock_auth_module
sys.modules['app.feedback'] = mock_feedback_module
sys.modules['app.redis_client'] = mock_redis_client_module
sys.modules['app.drift_detection'] = mock_drift_detection_module
sys.modules['app.circuit_breaker'] = mock_circuit_breaker_module
# NOTE: app.analytics is NOT mocked - it doesn't depend on external services
# and test_analytics.py tests the real implementation
sys.modules['app.device_utils'] = mock_device_utils_module

# =============================================================================
# Now safe to import pytest and define fixtures
# =============================================================================

import pytest
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def mock_external_connections():
    """Mock external service connections at session scope.

    This prevents the application from attempting to connect to
    Redis, PostgreSQL, Qdrant, or Ollama during test collection.

    The mocks are applied to sys.modules BEFORE any app imports occur.
    """
    # Set environment variables for any late-binding config reads
    env_patches = {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "OLLAMA_HOST": "http://localhost:11434",
        "POSTGRES_HOST": "localhost",
        "AUTH_ENABLED": "false",
        "QUERY_LOGGING_ENABLED": "false",
        "EMBEDDING_CACHE_ENABLED": "false",
        "RERANKER_ENABLED": "false",
        "HYBRID_SEARCH_ENABLED": "false",
        "HYDE_ENABLED": "false",
        "WEB_SEARCH_ENABLED": "false",
        "ENABLE_RETRIEVAL_METRICS": "false",
    }

    with patch.dict(os.environ, env_patches):
        yield


@pytest.fixture(autouse=True)
def reset_rag_module_mocks():
    """Reset RAG module mocks before each test.

    This ensures each test gets fresh mocks and prevents test pollution.
    """
    # Reset the mock call counts
    mock_vector_store_instance.reset_mock()
    mock_rag_pipeline_instance.reset_mock()

    # Re-setup default return values after reset for vector_store
    mock_vector_store_instance.is_connected.return_value = True
    mock_vector_store_instance.get_stats.return_value = {
        "collection_name": "test_collection",
        "vectors_count": 1000,
        "points_count": 1000,
    }

    # Reset search method side effects
    def mock_search_with_cache_info(query, top_k=5, min_score=0.3, source_type=None, source=None):
        return [], False

    def mock_hybrid_search_with_cache_info(query, top_k=5, min_score=0.3, source_type=None, source=None, alpha=0.5):
        return [], False

    mock_vector_store_instance.search_with_cache_info = MagicMock(side_effect=mock_search_with_cache_info)
    mock_vector_store_instance.hybrid_search_with_cache_info = MagicMock(side_effect=mock_hybrid_search_with_cache_info)

    # Re-setup default return values after reset for rag_pipeline
    mock_rag_pipeline_instance.is_ollama_connected.return_value = True
    mock_rag_pipeline_instance.list_models.return_value = [
        {"name": "llama3.1:8b", "size": "4.7GB", "modified_at": "2024-01-01"},
        {"name": "mistral:7b", "size": "4.1GB", "modified_at": "2024-01-01"},
    ]
    mock_rag_pipeline_instance.get_reranker_status.return_value = {
        "enabled": True,
        "loaded": True,
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }

    # Reset async generate_response mock
    async def mock_generate_response(*args, **kwargs):
        return {
            "response": "This is a test response about Kubernetes deployments.",
            "model": "llama3.1:8b",
            "context_used": True,
            "sources": [
                {
                    "source": "kubernetes/docs/deployments.md",
                    "source_type": "kubernetes",
                    "content_preview": "A Deployment provides declarative updates...",
                    "similarity_score": 0.89
                }
            ],
            "retrieval_metrics": {
                "initial_candidates": 20,
                "after_reranking": 5,
                "reranker_used": True,
                "avg_similarity_score": 0.85,
                "retrieval_time_ms": 45.2,
                "rerank_time_ms": 12.3
            }
        }

    mock_rag_pipeline_instance.generate_response = AsyncMock(side_effect=mock_generate_response)

    yield


@pytest.fixture
def mock_async_db_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    return session


@pytest.fixture
def sample_chat_response():
    """Sample chat response for testing."""
    return {
        "response": "Here is information about Kubernetes deployments...",
        "model": "llama3.1:8b",
        "context_used": True,
        "sources": [
            {
                "source": "kubernetes/concepts/workloads/controllers/deployment.md",
                "source_type": "kubernetes",
                "content_preview": "A Deployment provides declarative updates for Pods...",
                "similarity_score": 0.92,
                "rerank_score": 0.88,
                "rank": 1
            },
            {
                "source": "kubernetes/tasks/run-application/run-stateless-application-deployment.md",
                "source_type": "kubernetes",
                "content_preview": "This page shows how to run an application using...",
                "similarity_score": 0.87,
                "rerank_score": 0.82,
                "rank": 2
            }
        ],
        "retrieval_metrics": {
            "initial_candidates": 20,
            "after_reranking": 5,
            "reranker_used": True,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "hybrid_search_used": True,
            "hyde_used": False,
            "web_search_used": False,
            "avg_similarity_score": 0.85,
            "avg_rerank_score": 0.80,
            "retrieval_time_ms": 45.2,
            "rerank_time_ms": 12.3
        }
    }


@pytest.fixture
def sample_health_response():
    """Sample health response for testing."""
    return {
        "status": "healthy",
        "ollama_connected": True,
        "qdrant_connected": True,
        "redis_connected": True,
        "postgres_connected": True,
        "reranker_enabled": True,
        "reranker_loaded": True,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "redis_pool": {
            "max_connections": 10,
            "current_connections": 2,
            "available_connections": 8,
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "postgres_pool": {
            "host": "localhost",
            "port": 5432,
            "database": "devops_ai",
            "pool_size": 5,
            "max_overflow": 10,
            "checked_in": 5,
            "checked_out": 0,
            "overflow": 0,
            "pool_timeout": 30.0
        }
    }


@pytest.fixture
def sample_stats_response():
    """Sample vector store stats response."""
    return {
        "collection_name": "devops_docs",
        "vectors_count": 15432,
        "indexed_documents": 15432
    }


@pytest.fixture
def sample_models_response():
    """Sample models list response."""
    return {
        "models": [
            {
                "name": "llama3.1:8b",
                "size": "4.7GB",
                "modified": "2024-01-15T10:30:00Z"
            },
            {
                "name": "mistral:7b",
                "size": "4.1GB",
                "modified": "2024-01-10T08:00:00Z"
            },
            {
                "name": "qwen2.5-coder:7b",
                "size": "4.4GB",
                "modified": "2024-01-20T14:00:00Z"
            }
        ]
    }
