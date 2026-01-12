"""Test suite for FastAPI endpoints.

This module provides comprehensive tests for the DevOps AI Assistant API endpoints,
covering health checks, chat functionality, document upload, and analytics endpoints.
Uses pytest fixtures and mocking for external dependencies (Ollama, Qdrant, Redis, PostgreSQL).

NOTE: External module mocking (vectorstore, rag, etc.) is handled in conftest.py
which injects mocks into sys.modules before any app imports occur.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from io import BytesIO
import json

# Import mock instances from conftest for use in fixtures
from tests.conftest import mock_vector_store_instance, mock_rag_pipeline_instance


@pytest.fixture(scope="module")
def mock_settings():
    """Provide mocked settings for tests."""
    with patch("app.config.settings") as mock:
        mock.redis_host = "localhost"
        mock.redis_port = 6379
        mock.redis_db = 0
        mock.redis_max_connections = 10
        mock.redis_socket_timeout = 5
        mock.redis_socket_connect_timeout = 5
        mock.ollama_host = "http://localhost:11434"
        mock.ollama_default_model = "llama3.1:8b"
        mock.cors_origins_list = ["http://localhost:3000"]
        mock.query_logging_enabled = False
        mock.enable_retrieval_metrics = True
        mock.auth_enabled = False
        mock.postgres_pool_size = 5
        mock.postgres_max_overflow = 10
        mock.embedding_cache_enabled = False
        yield mock


@pytest.fixture(scope="module")
def mock_redis():
    """Mock Redis client and connection pool using the shared redis_client module."""
    mock_pool = MagicMock()
    mock_pool.max_connections = 10
    mock_pool._in_use_connections = set()
    mock_pool._available_connections = []

    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.lrange.return_value = []
    mock_client.pipeline.return_value = MagicMock()

    with patch("app.redis_client.get_redis_string_pool", return_value=mock_pool), \
         patch("app.redis_client.get_redis_pool", return_value=mock_pool), \
         patch("app.redis_client.get_redis_string_client", return_value=mock_client), \
         patch("app.redis_client.get_redis_client", return_value=mock_client), \
         patch("app.redis_client.is_redis_connected", return_value=True), \
         patch("app.redis_client.get_redis_pool_stats", return_value={
             "host": "localhost",
             "port": 6379,
             "db": 0,
             "max_connections": 10,
             "current_connections": 0,
             "available_connections": 5,
         }):
        yield mock_client


@pytest.fixture(scope="module")
def mock_rag_pipeline():
    """Mock RAG pipeline for chat tests - returns module-level mock from conftest."""
    with patch("app.main.rag_pipeline", mock_rag_pipeline_instance):
        yield mock_rag_pipeline_instance


@pytest.fixture(scope="module")
def mock_vector_store():
    """Mock vector store for stats tests - returns module-level mock from conftest."""
    with patch("app.main.vector_store", mock_vector_store_instance):
        yield mock_vector_store_instance


@pytest.fixture(scope="module")
def mock_database():
    """Mock PostgreSQL database connections."""
    with patch("app.main.check_postgres_connection") as mock_check, \
         patch("app.main.get_postgres_pool_stats") as mock_stats, \
         patch("app.main.get_db_context") as mock_context, \
         patch("app.main.init_db") as mock_init, \
         patch("app.main.close_db") as mock_close:
        mock_check.return_value = True
        mock_stats.return_value = {
            "host": "localhost",
            "port": 5432,
            "database": "devops_ai",
            "pool_size": 5,
            "max_overflow": 10,
            "checked_in": 5,
            "checked_out": 0,
            "overflow": 0,
            "pool_timeout": 30
        }
        mock_init.return_value = None
        mock_close.return_value = None

        # Mock async context manager for database sessions
        mock_db = AsyncMock()
        mock_context.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        mock_context.return_value.__aexit__ = AsyncMock(return_value=None)

        yield mock_db


@pytest.fixture(scope="module")
def client(mock_settings, mock_redis, mock_rag_pipeline, mock_vector_store, mock_database):
    """Create test client with all dependencies mocked."""
    # Import app after mocking to prevent actual connections
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoint:
    """Tests for the /api/health endpoint."""

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 status code."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_correct_structure(self, client):
        """Test that health response contains expected fields."""
        response = client.get("/api/health")
        data = response.json()

        # Check required fields exist
        assert "status" in data
        assert "ollama_connected" in data
        assert "qdrant_connected" in data
        assert "redis_connected" in data
        assert "postgres_connected" in data
        assert "reranker_enabled" in data
        assert "reranker_loaded" in data

    def test_health_status_values(self, client):
        """Test that health status reflects component states."""
        response = client.get("/api/health")
        data = response.json()

        # With all mocks returning connected, status should be healthy
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["ollama_connected"], bool)
        assert isinstance(data["qdrant_connected"], bool)
        assert isinstance(data["redis_connected"], bool)

    def test_health_includes_pool_stats(self, client):
        """Test that health response includes connection pool statistics."""
        response = client.get("/api/health")
        data = response.json()

        # Redis pool stats
        if data.get("redis_pool"):
            assert "max_connections" in data["redis_pool"]

        # PostgreSQL pool stats
        if data.get("postgres_pool"):
            assert "pool_size" in data["postgres_pool"]


# =============================================================================
# Chat Endpoint Tests
# =============================================================================

class TestChatEndpoint:
    """Tests for the /api/chat endpoint."""

    def test_chat_basic_request(self, client, mock_rag_pipeline):
        """Test basic chat request returns valid response."""
        response = client.post(
            "/api/chat",
            json={"message": "How do I create a Kubernetes deployment?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "response" in data
        assert "model" in data
        assert "session_id" in data
        assert "context_used" in data

    def test_chat_with_model_parameter(self, client, mock_rag_pipeline):
        """Test chat request with specific model."""
        response = client.post(
            "/api/chat",
            json={
                "message": "Explain Docker containers",
                "model": "mistral:7b"
            }
        )

        assert response.status_code == 200

    def test_chat_with_session_id(self, client, mock_rag_pipeline):
        """Test chat request with session ID for conversation continuity."""
        session_id = "test-session-12345"
        response = client.post(
            "/api/chat",
            json={
                "message": "What is Terraform?",
                "session_id": session_id
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    def test_chat_with_temperature(self, client, mock_rag_pipeline):
        """Test chat request with temperature parameter."""
        response = client.post(
            "/api/chat",
            json={
                "message": "Describe CI/CD pipelines",
                "temperature": 0.5
            }
        )

        assert response.status_code == 200

    def test_chat_without_rag(self, client, mock_rag_pipeline):
        """Test chat request with RAG disabled."""
        response = client.post(
            "/api/chat",
            json={
                "message": "What is DevOps?",
                "use_rag": False
            }
        )

        assert response.status_code == 200

    def test_chat_returns_sources(self, client, mock_rag_pipeline):
        """Test that chat response includes source documents."""
        response = client.post(
            "/api/chat",
            json={"message": "How do I configure nginx?"}
        )

        assert response.status_code == 200
        data = response.json()

        # Sources should be present when context_used is True
        if data.get("context_used"):
            assert "sources" in data

    def test_chat_returns_retrieval_metrics(self, client, mock_rag_pipeline):
        """Test that chat response includes retrieval metrics."""
        response = client.post(
            "/api/chat",
            json={"message": "Explain Prometheus metrics"}
        )

        assert response.status_code == 200
        data = response.json()

        if data.get("retrieval_metrics"):
            metrics = data["retrieval_metrics"]
            assert "initial_candidates" in metrics or "reranker_used" in metrics

    def test_chat_missing_message_returns_422(self, client):
        """Test that missing message field returns validation error."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_invalid_temperature_returns_422(self, client):
        """Test that invalid temperature returns validation error."""
        response = client.post(
            "/api/chat",
            json={
                "message": "Test",
                "temperature": 3.0  # Max is 2.0
            }
        )
        assert response.status_code == 422


# =============================================================================
# Document Upload Endpoint Tests
# =============================================================================

class TestUploadEndpoint:
    """Tests for the /api/upload endpoint."""

    def test_upload_markdown_file(self, client):
        """Test uploading a markdown file."""
        content = b"# Test Document\n\nThis is test content for the RAG system."
        files = {"files": ("test.md", BytesIO(content), "text/markdown")}

        # Mock the file system operations
        with patch("app.main.Path.mkdir"), \
             patch("builtins.open", MagicMock()), \
             patch("app.main.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            response = client.post(
                "/api/upload",
                files=files,
                data={"auto_ingest": "false"}  # Disable auto-ingest for test
            )

        assert response.status_code == 200
        data = response.json()
        assert "uploaded" in data
        assert "failed" in data

    def test_upload_text_file(self, client):
        """Test uploading a text file."""
        content = b"Plain text documentation content."
        files = {"files": ("readme.txt", BytesIO(content), "text/plain")}

        with patch("app.main.Path.mkdir"), \
             patch("builtins.open", MagicMock()):
            response = client.post(
                "/api/upload",
                files=files,
                data={"auto_ingest": "false"}
            )

        assert response.status_code == 200

    def test_upload_rejects_invalid_file_type(self, client):
        """Test that uploading unsupported file types is rejected."""
        content = b"Binary content"
        files = {"files": ("script.py", BytesIO(content), "text/x-python")}

        with patch("app.main.Path.mkdir"), \
             patch("builtins.open", MagicMock()):
            response = client.post(
                "/api/upload",
                files=files,
                data={"auto_ingest": "false"}
            )

        assert response.status_code == 200
        data = response.json()
        # The file should be in the failed list
        assert data["failed"] > 0 or (data["errors"] and len(data["errors"]) > 0)

    def test_upload_multiple_files(self, client):
        """Test uploading multiple files at once."""
        files = [
            ("files", ("doc1.md", BytesIO(b"# Doc 1"), "text/markdown")),
            ("files", ("doc2.md", BytesIO(b"# Doc 2"), "text/markdown")),
        ]

        with patch("app.main.Path.mkdir"), \
             patch("builtins.open", MagicMock()):
            response = client.post(
                "/api/upload",
                files=files,
                data={"auto_ingest": "false"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["uploaded"] == 2


# =============================================================================
# Analytics Endpoint Tests
# =============================================================================

class TestAnalyticsEndpoints:
    """Tests for analytics endpoints."""

    def test_stats_endpoint(self, client, mock_vector_store):
        """Test vector database stats endpoint."""
        response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()

        assert "collection_name" in data
        assert "vectors_count" in data
        assert "indexed_documents" in data

    def test_models_endpoint(self, client, mock_rag_pipeline):
        """Test models listing endpoint."""
        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)

    def test_templates_endpoint(self, client):
        """Test prompt templates endpoint."""
        with patch("app.main.get_templates") as mock_templates, \
             patch("app.main.get_categories") as mock_categories:
            mock_templates.return_value = [
                {
                    "id": "k8s-deployment",
                    "category": "Kubernetes",
                    "title": "Create Deployment",
                    "description": "Help with Kubernetes deployments",
                    "prompt": "How do I create a deployment for {app_name}?"
                }
            ]
            mock_categories.return_value = ["Kubernetes", "Docker", "Terraform"]

            response = client.get("/api/templates")

        assert response.status_code == 200
        data = response.json()

        assert "templates" in data
        assert "categories" in data

    def test_templates_filter_by_category(self, client):
        """Test filtering templates by category."""
        with patch("app.main.get_templates") as mock_templates, \
             patch("app.main.get_categories") as mock_categories:
            mock_templates.return_value = [
                {"id": "k8s-1", "category": "Kubernetes", "title": "Test", "description": "", "prompt": ""}
            ]
            mock_categories.return_value = ["Kubernetes"]

            response = client.get("/api/templates?category=Kubernetes")

        assert response.status_code == 200

    def test_history_endpoint(self, client, mock_redis):
        """Test conversation history endpoint."""
        mock_redis.lrange.return_value = [
            json.dumps({"role": "user", "content": "Hello"}),
            json.dumps({"role": "assistant", "content": "Hi there!"})
        ]

        response = client.get("/api/history/test-session-123")

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert "messages" in data


# =============================================================================
# Feedback Endpoint Tests
# =============================================================================

class TestFeedbackEndpoint:
    """Tests for the feedback endpoint."""

    def test_feedback_positive(self, client):
        """Test submitting positive feedback."""
        with patch("app.main.feedback_log") as mock_log, \
             patch("app.main.save_feedback_to_postgres") as mock_pg:
            mock_log.log_feedback.return_value = MagicMock(
                feedback_id="fb-123",
                timestamp="2024-01-01T00:00:00Z"
            )
            mock_pg.return_value = "fb-123"

            response = client.post(
                "/api/feedback",
                json={
                    "session_id": "test-session",
                    "helpful": True
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "saved"
        assert "feedback_id" in data

    def test_feedback_negative(self, client):
        """Test submitting negative feedback."""
        with patch("app.main.feedback_log") as mock_log, \
             patch("app.main.save_feedback_to_postgres") as mock_pg:
            mock_log.log_feedback.return_value = MagicMock(
                feedback_id="fb-456",
                timestamp="2024-01-01T00:00:00Z"
            )
            mock_pg.return_value = "fb-456"

            response = client.post(
                "/api/feedback",
                json={
                    "session_id": "test-session",
                    "helpful": False,
                    "message_index": 0
                }
            )

        assert response.status_code == 200

    def test_feedback_with_query_hash(self, client):
        """Test feedback with query hash for correlation."""
        with patch("app.main.feedback_log") as mock_log, \
             patch("app.main.save_feedback_to_postgres") as mock_pg:
            mock_log.log_feedback.return_value = MagicMock(
                feedback_id="fb-789",
                timestamp="2024-01-01T00:00:00Z"
            )
            mock_pg.return_value = "fb-789"

            response = client.post(
                "/api/feedback",
                json={
                    "session_id": "test-session",
                    "helpful": True,
                    "query_hash": "abc123def456"
                }
            )

        assert response.status_code == 200


# =============================================================================
# Root Endpoint Test
# =============================================================================

class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "docs" in data
        assert "health" in data


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json_returns_422(self, client):
        """Test that invalid JSON returns validation error."""
        response = client.post(
            "/api/chat",
            content="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_nonexistent_template_returns_404(self, client):
        """Test that requesting non-existent template returns 404."""
        with patch("app.main.get_template_by_id") as mock:
            mock.return_value = None

            response = client.get("/api/templates/nonexistent-id")

        assert response.status_code == 404


# =============================================================================
# Integration-style Tests (with more complete mocking)
# =============================================================================

class TestChatIntegration:
    """Integration-style tests for chat workflow."""

    def test_chat_workflow_complete(self, client, mock_rag_pipeline, mock_redis):
        """Test complete chat workflow from request to response."""
        # First request
        response1 = client.post(
            "/api/chat",
            json={
                "message": "What is Kubernetes?",
                "session_id": "integration-test"
            }
        )

        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["session_id"] == "integration-test"

        # Follow-up request in same session
        response2 = client.post(
            "/api/chat",
            json={
                "message": "How do I deploy to it?",
                "session_id": "integration-test"
            }
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == "integration-test"
