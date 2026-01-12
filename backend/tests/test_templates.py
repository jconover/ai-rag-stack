"""Test suite for prompt template variable support.

Tests template rendering, variable validation, and API endpoints for
the dynamic template customization feature.
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.templates import (
    get_templates,
    get_template_by_id,
    render_template,
    render_template_by_id,
    validate_template_variables,
    _extract_variables_from_prompt,
)


# =============================================================================
# Unit Tests for Template Functions
# =============================================================================

class TestExtractVariables:
    """Tests for variable extraction from prompt strings."""

    def test_extract_single_variable(self):
        """Test extracting a single variable."""
        prompt = "Deploy {app_name} to production"
        variables = _extract_variables_from_prompt(prompt)
        assert variables == ["app_name"]

    def test_extract_multiple_variables(self):
        """Test extracting multiple variables."""
        prompt = "Deploy {app_name} with {replicas} replicas to {namespace}"
        variables = _extract_variables_from_prompt(prompt)
        assert set(variables) == {"app_name", "replicas", "namespace"}

    def test_extract_no_variables(self):
        """Test prompt with no variables."""
        prompt = "This is a static prompt"
        variables = _extract_variables_from_prompt(prompt)
        assert variables == []

    def test_ignore_escaped_braces(self):
        """Test that escaped braces are not treated as variables."""
        prompt = "Use {{json}} format with {variable}"
        variables = _extract_variables_from_prompt(prompt)
        # The {{json}} should not be extracted, only {variable}
        assert "variable" in variables
        assert "json" not in variables

    def test_variable_naming_rules(self):
        """Test that only valid Python identifiers are extracted."""
        prompt = "{valid_name} and {123invalid}"
        variables = _extract_variables_from_prompt(prompt)
        assert "valid_name" in variables
        # {123invalid} should not match because it starts with number


class TestValidateTemplateVariables:
    """Tests for template variable validation."""

    def test_validate_all_required_provided(self):
        """Test validation passes when all required variables are provided."""
        template = {
            "variables": [
                {"name": "app_name", "required": True},
                {"name": "replicas", "required": False, "default": 3}
            ]
        }
        provided = {"app_name": "my-app"}

        is_valid, missing, unknown = validate_template_variables(template, provided)

        assert is_valid is True
        assert missing == []

    def test_validate_missing_required(self):
        """Test validation fails when required variables are missing."""
        template = {
            "variables": [
                {"name": "app_name", "required": True},
                {"name": "namespace", "required": True}
            ]
        }
        provided = {"app_name": "my-app"}

        is_valid, missing, unknown = validate_template_variables(template, provided)

        assert is_valid is False
        assert "namespace" in missing

    def test_validate_unknown_variables(self):
        """Test that unknown variables are detected."""
        template = {
            "variables": [
                {"name": "app_name", "required": False}
            ]
        }
        provided = {"app_name": "my-app", "unknown_var": "value"}

        is_valid, missing, unknown = validate_template_variables(template, provided)

        assert is_valid is True  # Unknown vars don't make it invalid
        assert "unknown_var" in unknown

    def test_validate_empty_template_variables(self):
        """Test validation with template that has no defined variables."""
        template = {"variables": []}
        provided = {"some_var": "value"}

        is_valid, missing, unknown = validate_template_variables(template, provided)

        assert is_valid is True
        assert "some_var" in unknown


class TestRenderTemplate:
    """Tests for template rendering with variable substitution."""

    def test_render_with_provided_values(self):
        """Test rendering with all variables provided."""
        template = {
            "prompt": "Deploy {app_name} with {replicas} replicas",
            "variables": [
                {"name": "app_name", "required": True},
                {"name": "replicas", "type": "number", "required": False, "default": 3}
            ]
        }
        variables = {"app_name": "my-service", "replicas": 5}

        rendered, used, missing = render_template(template, variables)

        assert rendered == "Deploy my-service with 5 replicas"
        assert used["app_name"] == "my-service"
        assert used["replicas"] == 5
        assert missing == []

    def test_render_with_defaults(self):
        """Test rendering using default values."""
        template = {
            "prompt": "Deploy {app_name} with {replicas} replicas",
            "variables": [
                {"name": "app_name", "required": True},
                {"name": "replicas", "type": "number", "required": False, "default": 3}
            ]
        }
        variables = {"app_name": "my-service"}

        rendered, used, missing = render_template(template, variables)

        assert rendered == "Deploy my-service with 3 replicas"
        assert used["replicas"] == 3

    def test_render_with_placeholder_for_missing(self):
        """Test rendering uses placeholders for missing optional variables."""
        template = {
            "prompt": "Deploy {app_name} to {namespace}",
            "variables": [
                {"name": "app_name", "required": False},
                {"name": "namespace", "required": False}
            ]
        }
        variables = {}

        rendered, used, missing = render_template(template, variables)

        assert rendered == "Deploy <app_name> to <namespace>"

    def test_render_tracks_missing_required(self):
        """Test that missing required variables are tracked."""
        template = {
            "prompt": "Deploy {app_name}",
            "variables": [
                {"name": "app_name", "required": True}
            ]
        }
        variables = {}

        rendered, used, missing = render_template(template, variables, strict=False)

        assert "app_name" in missing
        assert "<app_name>" in rendered

    def test_render_strict_mode_raises_error(self):
        """Test that strict mode raises error for missing required variables."""
        template = {
            "prompt": "Deploy {app_name}",
            "variables": [
                {"name": "app_name", "required": True}
            ]
        }
        variables = {}

        with pytest.raises(ValueError, match="Required variable"):
            render_template(template, variables, strict=True)

    def test_render_type_coercion_number(self):
        """Test that number type variables are coerced."""
        template = {
            "prompt": "Use {replicas} replicas",
            "variables": [
                {"name": "replicas", "type": "number", "required": False}
            ]
        }
        variables = {"replicas": "5"}

        rendered, used, missing = render_template(template, variables)

        assert used["replicas"] == 5
        assert isinstance(used["replicas"], int)

    def test_render_type_coercion_boolean(self):
        """Test that boolean type variables are coerced."""
        template = {
            "prompt": "Enable feature: {enabled}",
            "variables": [
                {"name": "enabled", "type": "boolean", "required": False}
            ]
        }
        variables = {"enabled": "true"}

        rendered, used, missing = render_template(template, variables)

        assert used["enabled"] is True

    def test_render_handles_undefined_prompt_variables(self):
        """Test handling of variables in prompt not defined in variables list."""
        template = {
            "prompt": "Deploy {app_name} to {undefined_var}",
            "variables": [
                {"name": "app_name", "required": False, "default": "app"}
            ]
        }
        variables = {}

        rendered, used, missing = render_template(template, variables)

        assert "<undefined_var>" in rendered


class TestRenderTemplateById:
    """Tests for render_template_by_id function."""

    def test_render_existing_template(self):
        """Test rendering an existing template by ID."""
        result = render_template_by_id("k8s-debug-pod", {"pod_name": "my-pod"})

        assert result is not None
        rendered, used, missing = result
        assert "my-pod" in rendered
        assert used["pod_name"] == "my-pod"

    def test_render_nonexistent_template(self):
        """Test rendering a non-existent template returns None."""
        result = render_template_by_id("nonexistent-id", {})
        assert result is None


class TestPromptTemplates:
    """Tests for the PROMPT_TEMPLATES data structure."""

    def test_all_templates_have_required_fields(self):
        """Test that all templates have required fields."""
        templates = get_templates()

        for template in templates:
            assert "id" in template, f"Template missing 'id'"
            assert "category" in template, f"Template {template.get('id')} missing 'category'"
            assert "title" in template, f"Template {template.get('id')} missing 'title'"
            assert "description" in template, f"Template {template.get('id')} missing 'description'"
            assert "prompt" in template, f"Template {template.get('id')} missing 'prompt'"

    def test_all_templates_have_variables_field(self):
        """Test that all templates have a variables field."""
        templates = get_templates()

        for template in templates:
            assert "variables" in template, f"Template {template['id']} missing 'variables'"
            assert isinstance(template["variables"], list)

    def test_variable_definitions_are_valid(self):
        """Test that all variable definitions have required fields."""
        templates = get_templates()

        for template in templates:
            for var in template.get("variables", []):
                assert "name" in var, f"Variable in {template['id']} missing 'name'"
                # Type should default to string if not specified
                var_type = var.get("type", "string")
                assert var_type in ["string", "number", "boolean", "select"], \
                    f"Invalid type '{var_type}' in {template['id']}"

    def test_select_type_has_options(self):
        """Test that select type variables have options defined."""
        templates = get_templates()

        for template in templates:
            for var in template.get("variables", []):
                if var.get("type") == "select":
                    assert "options" in var and len(var["options"]) > 0, \
                        f"Select variable {var['name']} in {template['id']} missing options"


# =============================================================================
# API Endpoint Tests
# =============================================================================

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
    """Mock Redis client."""
    from unittest.mock import MagicMock
    with patch("app.main.redis_pool") as mock_pool, \
         patch("app.main.redis_client") as mock_client:
        mock_pool.max_connections = 10
        mock_pool._in_use_connections = set()
        mock_pool._available_connections = []
        mock_client.ping.return_value = True
        mock_client.lrange.return_value = []
        yield mock_client


@pytest.fixture(scope="module")
def mock_database():
    """Mock PostgreSQL database connections."""
    from unittest.mock import AsyncMock
    with patch("app.main.check_postgres_connection") as mock_check, \
         patch("app.main.get_postgres_pool_stats") as mock_stats, \
         patch("app.main.init_db") as mock_init, \
         patch("app.main.close_db") as mock_close:
        mock_check.return_value = True
        mock_stats.return_value = {"pool_size": 5, "max_overflow": 10}
        mock_init.return_value = None
        mock_close.return_value = None
        yield


@pytest.fixture(scope="module")
def client(mock_settings, mock_redis, mock_database):
    """Create test client with dependencies mocked."""
    # Import mocks from conftest
    from tests.conftest import mock_vector_store_instance, mock_rag_pipeline_instance

    with patch("app.main.vector_store", mock_vector_store_instance), \
         patch("app.main.rag_pipeline", mock_rag_pipeline_instance):
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client


class TestTemplatesEndpoint:
    """Tests for the /api/templates endpoints."""

    def test_list_templates_includes_variables(self, client):
        """Test that listing templates includes variable definitions."""
        response = client.get("/api/templates")

        assert response.status_code == 200
        data = response.json()

        assert "templates" in data
        for template in data["templates"]:
            assert "variables" in template

    def test_get_template_includes_variables(self, client):
        """Test that getting a single template includes variable definitions."""
        response = client.get("/api/templates/k8s-debug-pod")

        assert response.status_code == 200
        data = response.json()

        assert "variables" in data
        assert len(data["variables"]) > 0

        # Check first variable structure
        var = data["variables"][0]
        assert "name" in var
        assert "type" in var


class TestRenderTemplateEndpoint:
    """Tests for the /api/templates/render endpoint."""

    def test_render_template_basic(self, client):
        """Test basic template rendering."""
        response = client.post(
            "/api/templates/render",
            json={
                "template_id": "k8s-debug-pod",
                "variables": {"pod_name": "my-test-pod", "namespace": "production"}
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["template_id"] == "k8s-debug-pod"
        assert "my-test-pod" in data["rendered_prompt"]
        assert "production" in data["rendered_prompt"]
        assert data["variables_used"]["pod_name"] == "my-test-pod"

    def test_render_template_with_defaults(self, client):
        """Test template rendering uses defaults for missing optional variables."""
        response = client.post(
            "/api/templates/render",
            json={
                "template_id": "k8s-debug-pod",
                "variables": {"pod_name": "my-pod"}  # namespace not provided
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should use default value for namespace
        assert data["variables_used"]["namespace"] == "default"
        assert "default" in data["rendered_prompt"]

    def test_render_template_missing_required(self, client):
        """Test that missing required variables are tracked."""
        response = client.post(
            "/api/templates/render",
            json={
                "template_id": "k8s-deployment",
                "variables": {}  # app_name is required
            }
        )

        assert response.status_code == 200
        data = response.json()

        # app_name is required and should be in missing_required
        assert "app_name" in data["missing_required"]

    def test_render_template_not_found(self, client):
        """Test rendering non-existent template returns 404."""
        response = client.post(
            "/api/templates/render",
            json={
                "template_id": "nonexistent-template",
                "variables": {}
            }
        )

        assert response.status_code == 404

    def test_render_template_empty_variables(self, client):
        """Test rendering with empty variables uses all defaults."""
        response = client.post(
            "/api/templates/render",
            json={
                "template_id": "docker-compose",
                "variables": {}
            }
        )

        assert response.status_code == 200
        data = response.json()

        # All variables should use defaults
        assert len(data["missing_required"]) == 0
        assert "PostgreSQL" in data["rendered_prompt"]  # default database

    def test_render_template_returns_original_prompt(self, client):
        """Test that response includes original prompt template."""
        response = client.post(
            "/api/templates/render",
            json={
                "template_id": "k8s-debug-pod",
                "variables": {"pod_name": "test"}
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "original_prompt" in data
        assert "{pod_name}" in data["original_prompt"]
        assert "{namespace}" in data["original_prompt"]

    def test_render_template_type_conversion(self, client):
        """Test that numeric values are properly handled."""
        response = client.post(
            "/api/templates/render",
            json={
                "template_id": "k8s-deployment",
                "variables": {
                    "app_name": "my-app",
                    "replicas": "5",  # String that should work
                    "probe_port": 8080
                }
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Replicas should be converted to int
        assert data["variables_used"]["replicas"] == 5
