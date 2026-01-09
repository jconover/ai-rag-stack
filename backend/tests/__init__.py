"""Backend test suite for DevOps AI Assistant.

This package contains tests organized by type:
- Unit tests: test_*.py files testing individual components
- Integration tests: test files marked with @pytest.mark.integration
- API tests: Tests for FastAPI endpoints using TestClient/AsyncClient

Run tests with:
    pytest                          # Run all tests
    pytest -m unit                  # Run only unit tests
    pytest -m "not slow"            # Skip slow tests
    pytest --cov=app               # With coverage
"""
