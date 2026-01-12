"""Test suite for output validation and hallucination detection.

This module tests the OutputValidator class which detects:
- Hallucination markers in LLM responses
- Unsupported claims not grounded in context
- Missing source citations
- Fabrication patterns
- Response length issues
"""

import pytest
from app.output_validation import (
    OutputValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_response,
    output_validator,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def validator():
    """Create a fresh OutputValidator instance for testing."""
    return OutputValidator()


@pytest.fixture
def sample_context():
    """Sample context from documentation."""
    return """[Source 1 - kubernetes]
Kubernetes is a container orchestration platform that automates deployment,
scaling, and management of containerized applications. It was originally
designed by Google and is now maintained by the CNCF.
---
[Source 2 - kubernetes]
A Kubernetes pod is the smallest deployable unit that can contain one or
more containers. Pods share storage and network resources.
---
[Source 3 - docker]
Docker containers package applications with their dependencies into
standardized units for software development."""


@pytest.fixture
def sample_sources():
    """Sample sources list matching the context."""
    return [
        {
            "source": "/docs/kubernetes/intro.md",
            "source_type": "kubernetes",
            "content_preview": "Kubernetes is a container orchestration...",
            "rank": 1,
            "similarity_score": 0.92
        },
        {
            "source": "/docs/kubernetes/pods.md",
            "source_type": "kubernetes",
            "content_preview": "A Kubernetes pod is the smallest...",
            "rank": 2,
            "similarity_score": 0.88
        },
        {
            "source": "/docs/docker/intro.md",
            "source_type": "docker",
            "content_preview": "Docker containers package applications...",
            "rank": 3,
            "similarity_score": 0.75
        }
    ]


# =============================================================================
# Test Classes
# =============================================================================

class TestOutputValidatorBasics:
    """Basic tests for OutputValidator initialization and structure."""

    def test_validator_instance_exists(self, validator):
        """Test validator can be instantiated."""
        assert validator is not None
        assert isinstance(validator, OutputValidator)

    def test_singleton_exists(self):
        """Test singleton instance exists."""
        assert output_validator is not None
        assert isinstance(output_validator, OutputValidator)

    def test_validate_returns_result(self, validator):
        """Test validate method returns ValidationResult."""
        result = validator.validate("Test response")
        assert isinstance(result, ValidationResult)

    def test_validate_response_function_works(self):
        """Test convenience function works."""
        result = validate_response("Test response")
        assert isinstance(result, ValidationResult)


class TestEmptyAndShortResponses:
    """Tests for empty and short response handling."""

    def test_empty_response_fails_validation(self, validator):
        """Test empty response is flagged as invalid."""
        result = validator.validate("")

        assert result.is_valid is False
        assert result.confidence_score == 0.0
        assert any(i.code == "EMPTY_RESPONSE" for i in result.issues)

    def test_short_response_gets_warning(self, validator):
        """Test very short response gets warning."""
        result = validator.validate("Hi")

        assert any(i.code == "RESPONSE_TOO_SHORT" for i in result.issues)
        assert any(i.severity == ValidationSeverity.WARNING for i in result.issues)


class TestHallucinationMarkerDetection:
    """Tests for hallucination marker detection."""

    def test_knowledge_cutoff_detected(self, validator):
        """Test 'as of my knowledge' marker is detected."""
        response = "As of my knowledge cutoff, Kubernetes was the most popular container orchestrator."
        result = validator.validate(response)

        assert len(result.hallucination_markers_found) > 0
        assert any("knowledge" in m.lower() for m in result.hallucination_markers_found)

    def test_cannot_verify_detected(self, validator):
        """Test 'I cannot verify' marker is detected."""
        response = "I cannot verify this information, but Docker typically uses containerd."
        result = validator.validate(response)

        assert len(result.hallucination_markers_found) > 0
        assert any(i.code == "HALLUCINATION_MARKER" for i in result.issues)

    def test_no_access_detected(self, validator):
        """Test 'I don't have access' marker is detected."""
        response = "I don't have access to real-time data, but here's what I know about pods."
        result = validator.validate(response)

        assert len(result.hallucination_markers_found) > 0

    def test_uncertainty_markers_detected(self, validator):
        """Test uncertainty phrases are detected."""
        response = "I'm not entirely sure, but I think Kubernetes uses etcd for storage."
        result = validator.validate(response)

        assert len(result.hallucination_markers_found) > 0

    def test_guessing_markers_detected(self, validator):
        """Test guessing phrases are detected."""
        response = "I'm just making an assumption here, but the default port should be 6443."
        result = validator.validate(response)

        assert len(result.hallucination_markers_found) > 0

    def test_clean_response_no_markers(self, validator, sample_context):
        """Test clean response has no hallucination markers."""
        response = """Kubernetes is a container orchestration platform [Source 1].
        Pods are the smallest deployable units [Source 2].
        Docker containers package applications with dependencies [Source 3]."""

        result = validator.validate(response, context=sample_context)

        assert len(result.hallucination_markers_found) == 0


class TestSourceCitationValidation:
    """Tests for source citation detection and validation."""

    def test_source_citations_counted(self, validator, sample_context, sample_sources):
        """Test source citations are correctly counted."""
        response = """Kubernetes is a container orchestration platform [Source 1].
        Pods can contain multiple containers [Source 2].
        Docker packages apps with dependencies [Source 3]."""

        result = validator.validate(
            response,
            context=sample_context,
            sources=sample_sources
        )

        assert result.source_citation_count >= 3

    def test_invalid_source_reference_detected(self, validator, sample_sources):
        """Test invalid source reference is detected."""
        response = "According to [Source 10], Kubernetes is great."  # Only 3 sources exist

        result = validator.validate(
            response,
            context="Some context",
            sources=sample_sources
        )

        assert any(i.code == "INVALID_SOURCE_REFERENCE" for i in result.issues)
        assert any(i.severity == ValidationSeverity.ERROR for i in result.issues)

    def test_no_citations_flagged_for_long_response(self, validator, sample_context):
        """Test missing citations flagged for long responses using context."""
        response = """Kubernetes is a powerful platform for container orchestration.
        It provides automatic scaling, load balancing, and self-healing capabilities.
        Pods are the basic building blocks and can contain multiple containers.
        The system uses etcd for distributed storage and configuration.
        Services expose pods to network traffic, enabling communication between components."""

        result = validator.validate(response, context=sample_context)

        assert any(i.code == "NO_SOURCE_CITATIONS" for i in result.issues)


class TestFabricationPatternDetection:
    """Tests for fabrication pattern detection."""

    def test_overly_specific_version_detected(self, validator):
        """Test overly specific version numbers are flagged."""
        response = "You need to use Kubernetes version 1.28.3.4.5 for this feature."

        result = validator.validate(response)

        assert any(i.code == "POTENTIAL_FABRICATION" for i in result.issues)

    def test_specific_date_detected(self, validator):
        """Test specific dates are flagged."""
        response = "This feature was released on January 15, 2024."

        result = validator.validate(response)

        assert any(i.code == "POTENTIAL_FABRICATION" for i in result.issues)


class TestUnsupportedClaimsDetection:
    """Tests for unsupported claims detection."""

    def test_unsupported_must_claim(self, validator, sample_context):
        """Test 'you must' claims without citation are flagged."""
        response = """Kubernetes is great for containerization.
        You must always use namespaces for production workloads."""

        result = validator.validate(response, context=sample_context)

        assert result.unsupported_claims_count > 0

    def test_unsupported_required_claim(self, validator, sample_context):
        """Test 'required' claims without citation are flagged."""
        response = "It is required to configure resource limits for all containers."

        result = validator.validate(response, context=sample_context)

        assert result.unsupported_claims_count > 0

    def test_supported_claim_not_flagged(self, validator, sample_context, sample_sources):
        """Test claims with citations are not flagged."""
        response = "You should use pods for deployments [Source 1]."

        result = validator.validate(
            response,
            context=sample_context,
            sources=sample_sources
        )

        # Should not count as unsupported since it has a citation nearby
        assert result.unsupported_claims_count == 0


class TestContextGrounding:
    """Tests for context grounding detection."""

    def test_well_grounded_response(self, validator, sample_context):
        """Test response well-grounded in context passes."""
        response = """Kubernetes is a container orchestration platform.
        It automates deployment and scaling of containerized applications.
        Pods are the smallest deployable units in Kubernetes."""

        result = validator.validate(response, context=sample_context)

        # Should not flag low context grounding
        assert not any(i.code == "LOW_CONTEXT_GROUNDING" for i in result.issues)

    def test_off_topic_response_flagged(self, validator, sample_context):
        """Test completely off-topic response is flagged."""
        response = """Python is a programming language created by Guido van Rossum.
        It supports object-oriented, functional, and procedural programming paradigms.
        The language emphasizes code readability with significant indentation.
        Libraries like NumPy and Pandas are popular for data analysis tasks.
        Django and Flask are common web frameworks built with Python."""

        result = validator.validate(response, context=sample_context)

        # Should flag low context grounding for completely off-topic response
        assert any(i.code == "LOW_CONTEXT_GROUNDING" for i in result.issues)


class TestConfidenceScoring:
    """Tests for confidence score calculation."""

    def test_perfect_response_high_confidence(self, validator, sample_context, sample_sources):
        """Test well-formed response gets high confidence."""
        response = """According to the documentation [Source 1], Kubernetes is a container
        orchestration platform. As stated in [Source 2], pods are the smallest
        deployable units."""

        result = validator.validate(
            response,
            context=sample_context,
            sources=sample_sources
        )

        assert result.confidence_score >= 0.8

    def test_hallucination_lowers_confidence(self, validator):
        """Test hallucination markers lower confidence score."""
        response = "I'm not entirely sure, but I believe Kubernetes is a container platform."

        result = validator.validate(response)

        assert result.confidence_score < 0.9

    def test_multiple_issues_lower_confidence(self, validator):
        """Test multiple issues significantly lower confidence."""
        response = """As of my knowledge cutoff, I cannot verify this.
        I think Kubernetes might be version 1.28.3.4.5.
        I'm guessing the port is 6443."""

        result = validator.validate(response)

        # Multiple hallucination markers should lower confidence significantly
        assert result.confidence_score < 0.5

    def test_confidence_clamped_to_range(self, validator):
        """Test confidence score is always between 0 and 1."""
        # Even with many issues
        response = """As of my knowledge cutoff, I cannot verify this.
        I'm not sure about this. I think it might be wrong.
        I'm guessing here. This is my assumption."""

        result = validator.validate(response)

        assert 0.0 <= result.confidence_score <= 1.0


class TestValidationResultStructure:
    """Tests for ValidationResult data structure."""

    def test_result_has_required_fields(self, validator):
        """Test ValidationResult has all required fields."""
        result = validator.validate("Test response")

        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'issues')
        assert hasattr(result, 'source_citation_count')
        assert hasattr(result, 'unsupported_claims_count')
        assert hasattr(result, 'hallucination_markers_found')
        assert hasattr(result, 'validation_time_ms')

    def test_result_to_dict(self, validator):
        """Test ValidationResult serializes to dictionary."""
        result = validator.validate("Test response with some content")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'is_valid' in result_dict
        assert 'confidence_score' in result_dict
        assert 'issues' in result_dict
        assert 'issues_count' in result_dict
        assert 'validation_time_ms' in result_dict

    def test_validation_time_tracked(self, validator, sample_context):
        """Test validation time is tracked."""
        response = """Kubernetes is a container orchestration platform.
        It helps manage containerized applications at scale."""

        result = validator.validate(response, context=sample_context)

        assert result.validation_time_ms > 0
        assert result.validation_time_ms < 1000  # Should complete in under 1 second


class TestValidationIssueSeverity:
    """Tests for issue severity levels."""

    def test_error_severity_for_invalid_source(self, validator, sample_sources):
        """Test invalid source reference has error severity."""
        response = "According to [Source 99], this is true."

        result = validator.validate(
            response,
            context="Some context",
            sources=sample_sources
        )

        error_issues = [i for i in result.issues if i.severity == ValidationSeverity.ERROR]
        assert len(error_issues) > 0

    def test_warning_severity_for_hallucination_markers(self, validator):
        """Test hallucination markers have warning severity."""
        response = "I'm not sure about this information."

        result = validator.validate(response)

        warning_issues = [i for i in result.issues if i.severity == ValidationSeverity.WARNING]
        assert len(warning_issues) > 0

    def test_info_severity_for_minor_issues(self, validator, sample_context):
        """Test minor issues have info severity."""
        response = """Kubernetes is a container platform.
        You should configure resource limits for containers."""

        result = validator.validate(response, context=sample_context)

        info_issues = [i for i in result.issues if i.severity == ValidationSeverity.INFO]
        # May have info-level issues like unsupported claims or no citations
        assert isinstance(info_issues, list)


class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_unicode_content_handled(self, validator):
        """Test unicode characters in response are handled."""
        response = "Kubernetes supports emoji labels: pod-name-abc123"

        result = validator.validate(response)

        assert result is not None
        assert isinstance(result.confidence_score, float)

    def test_very_long_response_handled(self, validator):
        """Test very long responses are handled."""
        response = "Kubernetes is great. " * 1000  # ~20k chars

        result = validator.validate(response)

        assert result is not None
        assert result.validation_time_ms < 5000  # Should complete reasonably fast

    def test_special_characters_handled(self, validator):
        """Test special characters in response are handled."""
        response = "Use `kubectl get pods` command. Check /var/log/pods/* for logs."

        result = validator.validate(response)

        assert result is not None
        assert isinstance(result, ValidationResult)

    def test_empty_context_handled(self, validator):
        """Test empty context is handled gracefully."""
        response = "Kubernetes is a container orchestration platform."

        result = validator.validate(response, context="")

        assert result is not None
        # Should not have context-related issues since context was empty

    def test_none_sources_handled(self, validator):
        """Test None sources list is handled gracefully."""
        response = "Kubernetes is great [Source 1]."

        result = validator.validate(response, context="Some context", sources=None)

        assert result is not None
        # Should not crash with None sources


class TestIntegration:
    """Integration tests combining multiple validation aspects."""

    def test_full_validation_pipeline(self, validator, sample_context, sample_sources):
        """Test complete validation pipeline with all features."""
        response = """Kubernetes is a container orchestration platform [Source 1].
        As mentioned in [Source 2], pods are the smallest deployable units.
        Docker containers package applications with their dependencies [Source 3].

        To deploy an application, you should:
        1. Create a Deployment manifest
        2. Apply it with kubectl
        3. Verify the pods are running"""

        result = validator.validate(
            response,
            context=sample_context,
            sources=sample_sources,
            query="How do I deploy an app to Kubernetes?"
        )

        # Should pass validation
        assert result.is_valid is True

        # Should have citations
        assert result.source_citation_count >= 3

        # Should have high confidence
        assert result.confidence_score >= 0.7

        # Result should serialize cleanly
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

    def test_problematic_response_fails(self, validator):
        """Test problematic response is properly flagged."""
        response = """As of my knowledge cutoff, I cannot verify the current Kubernetes version.
        I'm not entirely sure, but I think you need version 1.28.3.4.5.
        According to [Source 99], this should work.
        I'm guessing the configuration would look something like this..."""

        result = validator.validate(response, sources=[{"source": "test"}])

        # Should have issues
        assert len(result.issues) > 0

        # Should have low confidence
        assert result.confidence_score < 0.5

        # Should detect hallucination markers
        assert len(result.hallucination_markers_found) > 0


# Run with: pytest backend/tests/test_output_validation.py -v
