"""Output validation and hallucination detection for LLM responses.

This module implements guardrails to detect potential hallucinations,
unsupported claims, and other quality issues in LLM-generated responses.

Key features:
- Hallucination marker detection (phrases indicating uncertainty)
- Source citation validation (claims should reference provided context)
- Confidence scoring based on source attribution
- Response length validation
- Pattern detection for fabricated content
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue detected in the response."""
    code: str  # Machine-readable issue code
    message: str  # Human-readable description
    severity: ValidationSeverity
    position: Optional[int] = None  # Character position in response
    snippet: Optional[str] = None  # Relevant text snippet


@dataclass
class ValidationResult:
    """Complete validation result for an LLM response."""
    is_valid: bool = True
    confidence_score: float = 1.0  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    source_citation_count: int = 0
    unsupported_claims_count: int = 0
    hallucination_markers_found: List[str] = field(default_factory=list)
    validation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "is_valid": self.is_valid,
            "confidence_score": round(self.confidence_score, 3),
            "issues_count": len(self.issues),
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity.value,
                    "snippet": issue.snippet[:100] + "..." if issue.snippet and len(issue.snippet) > 100 else issue.snippet
                }
                for issue in self.issues
            ],
            "source_citation_count": self.source_citation_count,
            "unsupported_claims_count": self.unsupported_claims_count,
            "hallucination_markers_found": self.hallucination_markers_found,
            "validation_time_ms": round(self.validation_time_ms, 2)
        }


class OutputValidator:
    """Validates LLM responses for hallucinations and quality issues."""

    # Hallucination markers - phrases indicating the model is uncertain or making things up
    HALLUCINATION_MARKERS = [
        # Knowledge limitation phrases
        r"as of my (knowledge|training|last update)",
        r"as of my cutoff",
        r"i don'?t have access to",
        r"i cannot (verify|confirm|access|check)",
        r"i'?m not able to (verify|confirm|access|check)",
        r"i don'?t have (current|real-?time|live|up-to-date) (information|data|access)",
        r"my (training|knowledge) (data|cutoff)",
        r"i cannot browse",
        r"i'?m unable to access",

        # Uncertainty indicators
        r"i believe (this|that|it) (may|might|could)",
        r"if i (recall|remember) correctly",
        r"i think (this|that|it) (should|would|could|might)",
        r"i'?m not (entirely |completely |100% )?sure",
        r"i'?m not certain",
        r"i cannot guarantee",
        r"this (may|might|could) (not )?be (accurate|correct|up-to-date)",

        # Admission of fabrication
        r"i'?m (just )?making (this|an) (assumption|guess)",
        r"i'?m guessing",
        r"this is (just )?my (assumption|speculation|guess)",
        r"i don'?t have (specific|exact|precise) (information|data|details)",
        r"i cannot find (any|specific) (information|documentation|reference)",

        # Hedging phrases that often precede hallucinations
        r"(typically|usually|generally),? (i would|one would|you would) expect",
        r"based on (common|general|typical) (practice|patterns|conventions)",
        r"(from|in) my (experience|understanding)",
    ]

    # Patterns that often indicate fabricated specifics
    FABRICATION_PATTERNS = [
        # Overly specific version numbers without citation
        r"version (\d+\.\d+\.\d+\.\d+)",  # Very specific versions like 1.2.3.4
        r"released (on|in) (january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2},? \d{4}",
        # Fake URLs (not from context)
        r"(visit|see|check|go to) (https?://[^\s]+)",
        # Suspiciously specific statistics
        r"(approximately|about|around|roughly) \d+\.?\d*%",
        # Phone numbers, emails (could be fabricated)
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone numbers
    ]

    # Source citation patterns
    SOURCE_CITATION_PATTERNS = [
        r"\[source\s*\d+\]",
        r"\[source\s+\d+\s*[-–]\s*[^\]]+\]",
        r"according to (source|the documentation|the context)",
        r"from the (provided |given )?(context|documentation|sources)",
        r"as (mentioned|stated|shown|described) in",
    ]

    # Claim indicators - phrases that introduce claims needing support
    CLAIM_INDICATORS = [
        r"you (must|should|need to|have to)",
        r"it is (required|necessary|mandatory|essential) to",
        r"the (correct|proper|right|best) way",
        r"always use",
        r"never use",
        r"this will (cause|result in|lead to)",
        r"by default,? (\w+ )*(is|are|will)",
        r"the (official|recommended|standard) (approach|method|way)",
    ]

    # Minimum and maximum response length thresholds
    MIN_RESPONSE_LENGTH = 10
    MAX_RESPONSE_LENGTH = 50000

    def __init__(self):
        # Compile regex patterns for performance
        self._hallucination_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.HALLUCINATION_MARKERS
        ]
        self._fabrication_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.FABRICATION_PATTERNS
        ]
        self._citation_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SOURCE_CITATION_PATTERNS
        ]
        self._claim_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CLAIM_INDICATORS
        ]

    def validate(
        self,
        response: str,
        context: str = "",
        sources: Optional[List[Dict[str, Any]]] = None,
        query: str = ""
    ) -> ValidationResult:
        """Validate an LLM response for potential issues.

        Args:
            response: The LLM-generated response text
            context: The context provided to the LLM
            sources: List of source documents used
            query: The original user query

        Returns:
            ValidationResult with detected issues and confidence score
        """
        import time
        start_time = time.perf_counter()

        result = ValidationResult()

        if not response:
            result.is_valid = False
            result.confidence_score = 0.0
            result.issues.append(ValidationIssue(
                code="EMPTY_RESPONSE",
                message="Response is empty",
                severity=ValidationSeverity.ERROR
            ))
            return result

        # Run all validation checks
        self._check_response_length(response, result)
        self._check_hallucination_markers(response, result)
        self._check_fabrication_patterns(response, result)
        self._check_source_citations(response, context, sources, result)
        self._check_unsupported_claims(response, context, result)
        self._check_context_grounding(response, context, result)

        # Calculate overall confidence score
        result.confidence_score = self._calculate_confidence_score(result, sources)

        # Determine overall validity
        error_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.ERROR)
        result.is_valid = error_count == 0

        result.validation_time_ms = (time.perf_counter() - start_time) * 1000

        # Log warnings for significant issues
        if result.hallucination_markers_found:
            logger.warning(
                f"Hallucination markers detected in response: {result.hallucination_markers_found}"
            )

        if result.confidence_score < 0.5:
            logger.warning(
                f"Low confidence response (score={result.confidence_score:.2f}): "
                f"{len(result.issues)} issues detected"
            )

        return result

    def _check_response_length(self, response: str, result: ValidationResult) -> None:
        """Check if response length is within acceptable bounds."""
        length = len(response)

        if length < self.MIN_RESPONSE_LENGTH:
            result.issues.append(ValidationIssue(
                code="RESPONSE_TOO_SHORT",
                message=f"Response is too short ({length} chars, minimum {self.MIN_RESPONSE_LENGTH})",
                severity=ValidationSeverity.WARNING
            ))

        if length > self.MAX_RESPONSE_LENGTH:
            result.issues.append(ValidationIssue(
                code="RESPONSE_TOO_LONG",
                message=f"Response exceeds maximum length ({length} chars, maximum {self.MAX_RESPONSE_LENGTH})",
                severity=ValidationSeverity.WARNING
            ))

    def _check_hallucination_markers(self, response: str, result: ValidationResult) -> None:
        """Detect phrases that often indicate hallucination or uncertainty."""
        for pattern in self._hallucination_patterns:
            matches = pattern.finditer(response)
            for match in matches:
                marker_text = match.group(0)
                if marker_text not in result.hallucination_markers_found:
                    result.hallucination_markers_found.append(marker_text)
                    result.issues.append(ValidationIssue(
                        code="HALLUCINATION_MARKER",
                        message=f"Detected uncertainty/hallucination marker: '{marker_text}'",
                        severity=ValidationSeverity.WARNING,
                        position=match.start(),
                        snippet=response[max(0, match.start()-20):match.end()+20]
                    ))

    def _check_fabrication_patterns(self, response: str, result: ValidationResult) -> None:
        """Detect patterns that often indicate fabricated content."""
        for pattern in self._fabrication_patterns:
            matches = pattern.finditer(response)
            for match in matches:
                result.issues.append(ValidationIssue(
                    code="POTENTIAL_FABRICATION",
                    message=f"Potentially fabricated detail: '{match.group(0)}'",
                    severity=ValidationSeverity.INFO,
                    position=match.start(),
                    snippet=response[max(0, match.start()-20):match.end()+20]
                ))

    def _check_source_citations(
        self,
        response: str,
        context: str,
        sources: Optional[List[Dict[str, Any]]],
        result: ValidationResult
    ) -> None:
        """Check for proper source citations in the response."""
        citation_count = 0
        for pattern in self._citation_patterns:
            citation_count += len(pattern.findall(response))

        result.source_citation_count = citation_count

        # If context was provided but no citations found, flag it
        if context and citation_count == 0 and len(response) > 200:
            result.issues.append(ValidationIssue(
                code="NO_SOURCE_CITATIONS",
                message="Response uses context but contains no source citations",
                severity=ValidationSeverity.INFO
            ))

        # Check if cited source numbers are valid
        source_refs = re.findall(r"\[source\s*(\d+)\]", response, re.IGNORECASE)
        if sources and source_refs:
            max_source = len(sources)
            for ref in source_refs:
                ref_num = int(ref)
                if ref_num < 1 or ref_num > max_source:
                    result.issues.append(ValidationIssue(
                        code="INVALID_SOURCE_REFERENCE",
                        message=f"Reference [Source {ref_num}] is invalid (only {max_source} sources available)",
                        severity=ValidationSeverity.ERROR,
                        snippet=f"[Source {ref_num}]"
                    ))

    def _check_unsupported_claims(
        self,
        response: str,
        context: str,
        result: ValidationResult
    ) -> None:
        """Check for claims that may not be supported by the context."""
        if not context:
            return

        # Find claim indicators in response
        claims_found = []
        for pattern in self._claim_patterns:
            matches = pattern.finditer(response)
            for match in matches:
                claims_found.append((match.start(), match.group(0)))

        # Check if claims have nearby citations
        for pos, claim in claims_found:
            # Look for citation within 200 chars after the claim
            following_text = response[pos:pos+200]
            has_citation = any(p.search(following_text) for p in self._citation_patterns)

            if not has_citation:
                result.unsupported_claims_count += 1
                result.issues.append(ValidationIssue(
                    code="UNSUPPORTED_CLAIM",
                    message=f"Claim may not be supported by context: '{claim}'",
                    severity=ValidationSeverity.INFO,
                    position=pos,
                    snippet=response[max(0, pos-10):pos+len(claim)+50]
                ))

    def _check_context_grounding(
        self,
        response: str,
        context: str,
        result: ValidationResult
    ) -> None:
        """Check if response is grounded in the provided context.

        Uses a simple but effective approach: extract key terms from context
        and check how many appear in the response.
        """
        if not context:
            return

        # Extract significant words (3+ chars, not common stopwords)
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'will', 'more',
            'when', 'who', 'which', 'their', 'said', 'each', 'she', 'how', 'this',
            'from', 'they', 'with', 'that', 'what', 'were', 'would', 'there', 'about',
            'into', 'than', 'them', 'could', 'only', 'other', 'then', 'these', 'also',
            'after', 'before', 'some', 'such', 'very', 'just', 'over', 'made', 'your',
            'source', 'unknown'
        }

        # Extract words from context
        context_words = set(
            word.lower() for word in re.findall(r'\b[a-zA-Z]{3,}\b', context)
            if word.lower() not in stopwords
        )

        # Extract words from response
        response_words = set(
            word.lower() for word in re.findall(r'\b[a-zA-Z]{3,}\b', response)
            if word.lower() not in stopwords
        )

        if not context_words or not response_words:
            return

        # Calculate overlap ratio
        overlap = response_words & context_words
        overlap_ratio = len(overlap) / len(response_words) if response_words else 0

        # If response has very low overlap with context, flag it
        if overlap_ratio < 0.1 and len(response_words) > 20:
            result.issues.append(ValidationIssue(
                code="LOW_CONTEXT_GROUNDING",
                message=f"Response has low overlap with context ({overlap_ratio:.1%} term overlap)",
                severity=ValidationSeverity.WARNING
            ))

    def _calculate_confidence_score(
        self,
        result: ValidationResult,
        sources: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Calculate overall confidence score based on validation results.

        Score factors:
        - Base score starts at 1.0
        - Deductions for issues based on severity
        - Bonus for proper source citations
        - Penalty for hallucination markers
        """
        score = 1.0

        # Deductions for issues
        for issue in result.issues:
            if issue.severity == ValidationSeverity.ERROR:
                score -= 0.25
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.10
            elif issue.severity == ValidationSeverity.INFO:
                score -= 0.02

        # Bonus for source citations (up to 0.1)
        if sources and result.source_citation_count > 0:
            citation_bonus = min(0.1, result.source_citation_count * 0.02)
            score += citation_bonus

        # Penalty for hallucination markers
        if result.hallucination_markers_found:
            score -= 0.15 * len(result.hallucination_markers_found)

        # Penalty for unsupported claims
        if result.unsupported_claims_count > 0:
            score -= 0.05 * min(result.unsupported_claims_count, 5)

        # Clamp to valid range
        return max(0.0, min(1.0, score))


# Singleton instance
output_validator = OutputValidator()


def validate_response(
    response: str,
    context: str = "",
    sources: Optional[List[Dict[str, Any]]] = None,
    query: str = ""
) -> ValidationResult:
    """Convenience function to validate a response.

    Args:
        response: The LLM-generated response text
        context: The context provided to the LLM
        sources: List of source documents used
        query: The original user query

    Returns:
        ValidationResult with detected issues and confidence score
    """
    return output_validator.validate(response, context, sources, query)
