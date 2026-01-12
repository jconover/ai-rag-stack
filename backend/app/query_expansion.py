"""HyDE (Hypothetical Document Embeddings) query expansion module.

HyDE improves retrieval quality by generating a hypothetical document that would
answer the user's query, then using that document's embedding for similarity search.
This bridges the semantic gap between short queries and longer document chunks.

Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)

Usage:
    from app.query_expansion import hyde_expander

    # Check if HyDE should be used for this query
    if hyde_expander.should_expand(query):
        hypothetical_doc = await hyde_expander.expand(query)
        # Use hypothetical_doc for embedding and retrieval
    else:
        # Use original query directly
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import ollama

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class HyDEConfig:
    """Configuration for HyDE query expansion.

    Configuration is loaded from the centralized settings in config.py,
    which reads from environment variables. This can be overridden by
    passing custom values to the constructor.
    """

    # Enable/disable HyDE
    enabled: bool = None

    # Model to use for generating hypothetical documents
    # Uses a smaller/faster model by default for lower latency
    model: str = None

    # Temperature for generation (lower = more focused/deterministic)
    temperature: float = None

    # Maximum tokens for the hypothetical document
    max_tokens: int = None

    # Query length thresholds for skipping HyDE
    min_query_length: int = None
    max_query_length: int = None

    # Timeout for HyDE generation in seconds
    timeout_seconds: float = None

    # Whether to cache hypothetical documents (future enhancement)
    cache_enabled: bool = False

    def __post_init__(self):
        """Load defaults from centralized settings if not explicitly set."""
        if self.enabled is None:
            self.enabled = settings.hyde_enabled
        if self.model is None:
            self.model = settings.hyde_model
        if self.temperature is None:
            self.temperature = settings.hyde_temperature
        if self.max_tokens is None:
            self.max_tokens = settings.hyde_max_tokens
        if self.min_query_length is None:
            self.min_query_length = settings.hyde_min_query_length
        if self.max_query_length is None:
            self.max_query_length = settings.hyde_max_query_length
        if self.timeout_seconds is None:
            self.timeout_seconds = settings.hyde_timeout_seconds


@dataclass
class HyDEResult:
    """Result container for HyDE expansion."""

    original_query: str
    hypothetical_document: Optional[str] = None
    expanded: bool = False
    skip_reason: Optional[str] = None
    generation_time_ms: float = 0.0
    model_used: Optional[str] = None
    error: Optional[str] = None


class HyDEExpander:
    """Hypothetical Document Embeddings (HyDE) query expander.

    HyDE generates a hypothetical document that would answer the query,
    improving retrieval by matching against document-like embeddings
    rather than short query embeddings.

    Features:
    - Intelligent query analysis to skip HyDE for already-specific queries
    - DevOps-focused prompt template
    - Configurable via environment variables
    - Async support for non-blocking operation
    - Detailed metrics and error handling

    Example:
        expander = HyDEExpander()

        # Vague query - HyDE will help
        result = await expander.expand("kubernetes networking")
        # result.hypothetical_document contains a detailed doc about K8s networking

        # Specific query - HyDE skipped
        result = await expander.expand("kubectl get pods -n kube-system")
        # result.expanded = False, result.skip_reason = "cli_command"
    """

    # Patterns that indicate the query is already specific enough
    SKIP_PATTERNS = {
        # Error messages and stack traces
        'error_message': re.compile(
            r'(error|exception|traceback|failed|cannot|unable|denied|refused|timeout|'
            r'connection refused|permission denied|not found|no such|invalid|'
            r'ENOENT|EACCES|ETIMEDOUT|OOMKilled|CrashLoopBackOff)',
            re.IGNORECASE
        ),

        # CLI commands (kubectl, docker, terraform, etc.)
        'cli_command': re.compile(
            r'^(kubectl|docker|terraform|ansible|helm|git|aws|gcloud|az|'
            r'systemctl|journalctl|curl|wget|ssh|scp|rsync|make|npm|pip|'
            r'podman|crictl|istioctl|argocd|flux)\s+',
            re.IGNORECASE
        ),

        # File paths
        'file_path': re.compile(
            r'(/[a-zA-Z0-9_.-]+){2,}|'
            r'[a-zA-Z]:\\\\|'
            r'\.(yaml|yml|json|tf|md|py|sh|conf|config)$',
            re.IGNORECASE
        ),

        # Configuration snippets (YAML, JSON-like)
        'config_snippet': re.compile(
            r'(apiVersion:|kind:|metadata:|spec:|name:|image:|ports:|'
            r'"[^"]+"\s*:\s*["\[{]|'
            r'^\s*-\s+\w+:)',
            re.IGNORECASE | re.MULTILINE
        ),

        # Log entries
        'log_entry': re.compile(
            r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}|'
            r'\[(INFO|WARN|ERROR|DEBUG|FATAL)\]|'
            r'level=(info|warn|error|debug)',
            re.IGNORECASE
        ),

        # IP addresses, ports, URLs
        'network_specific': re.compile(
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?|'
            r'https?://[^\s]+|'
            r':[0-9]{2,5}\b'
        ),

        # Resource names with specific identifiers
        'resource_identifier': re.compile(
            r'[a-z0-9]+-[a-z0-9]+-[a-z0-9]+|'  # UUID-like patterns
            r'sha256:[a-f0-9]{64}|'  # Container image digests
            r'[a-f0-9]{40}'  # Git commit hashes
        ),
    }

    # DevOps-focused prompt template for generating hypothetical documents
    # Optimized for retrieval quality: matches documentation style in vector store
    HYDE_PROMPT_TEMPLATE = """Write a 150-200 word technical documentation excerpt answering this DevOps question. Write in present tense, declarative style.

Requirements:
- Use specific technical terminology (e.g., "Pod", "ConfigMap", "terraform plan", "docker build")
- Include at least one concrete example: a CLI command, YAML snippet, or configuration block
- Reference actual tool names, flags, file paths, or environment variables
- State facts directly without hedging ("X does Y" not "X can help with Y")

Forbidden:
- Meta-commentary ("This document...", "In this guide...", "Let me explain...")
- Filler phrases ("It's important to note...", "One thing to consider...")
- Rhetorical questions
- First/second person ("I", "you", "we")

Domain focus: Kubernetes, Docker, Terraform, Ansible, Helm, AWS/GCP/Azure, CI/CD pipelines, monitoring (Prometheus/Grafana), GitOps, Linux administration.

Question: {query}

Documentation:"""

    def __init__(self, config: Optional[HyDEConfig] = None):
        """Initialize the HyDE expander.

        Args:
            config: Optional HyDEConfig instance. If not provided,
                   configuration is loaded from environment variables.
        """
        self.config = config or HyDEConfig()
        self._ollama_client = None

    @property
    def enabled(self) -> bool:
        """Check if HyDE is enabled."""
        return self.config.enabled

    def should_expand(self, query: str) -> tuple[bool, Optional[str]]:
        """Determine if the query should be expanded using HyDE.

        Analyzes the query to determine if HyDE would be beneficial.
        Returns False for queries that are already specific enough,
        such as error messages, CLI commands, or configuration snippets.

        Args:
            query: The user's search query

        Returns:
            Tuple of (should_expand: bool, skip_reason: Optional[str])
        """
        if not self.config.enabled:
            return False, "hyde_disabled"

        query = query.strip()

        # Check query length bounds
        if len(query) < self.config.min_query_length:
            return False, "query_too_short"

        if len(query) > self.config.max_query_length:
            return False, "query_too_long"

        # Check for patterns that indicate the query is already specific
        for pattern_name, pattern in self.SKIP_PATTERNS.items():
            if pattern.search(query):
                logger.debug(f"HyDE skipped: query matches '{pattern_name}' pattern")
                return False, pattern_name

        # Count the number of words - very short queries benefit most from HyDE
        word_count = len(query.split())

        # Queries with many words are usually already specific
        if word_count > 30:
            return False, "too_many_words"

        return True, None

    def _build_prompt(self, query: str) -> str:
        """Build the prompt for hypothetical document generation.

        Args:
            query: The user's search query

        Returns:
            Formatted prompt string
        """
        return self.HYDE_PROMPT_TEMPLATE.format(query=query)

    def _generate_sync(self, query: str) -> HyDEResult:
        """Synchronously generate a hypothetical document.

        Args:
            query: The user's search query

        Returns:
            HyDEResult with the hypothetical document or error
        """
        result = HyDEResult(original_query=query)

        # Check if we should expand this query
        should_expand, skip_reason = self.should_expand(query)
        if not should_expand:
            result.skip_reason = skip_reason
            return result

        start_time = time.perf_counter()

        try:
            prompt = self._build_prompt(query)

            response = ollama.generate(
                model=self.config.model,
                prompt=prompt,
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_tokens,
                }
            )

            hypothetical_doc = response.get('response', '').strip()

            if hypothetical_doc:
                result.hypothetical_document = hypothetical_doc
                result.expanded = True
                result.model_used = self.config.model
            else:
                result.error = "Empty response from model"

        except Exception as e:
            result.error = str(e)
            logger.error(f"HyDE generation failed: {e}")

        result.generation_time_ms = (time.perf_counter() - start_time) * 1000

        if result.expanded:
            logger.debug(
                f"HyDE expanded query in {result.generation_time_ms:.1f}ms: "
                f"'{query[:50]}...' -> {len(result.hypothetical_document)} chars"
            )

        return result

    async def expand(self, query: str) -> HyDEResult:
        """Asynchronously generate a hypothetical document for the query.

        This method runs the synchronous Ollama call in a thread pool
        to avoid blocking the event loop.

        Args:
            query: The user's search query

        Returns:
            HyDEResult with the hypothetical document or error/skip info
        """
        result = HyDEResult(original_query=query)

        # Check if we should expand this query (quick check before async work)
        should_expand, skip_reason = self.should_expand(query)
        if not should_expand:
            result.skip_reason = skip_reason
            return result

        # Run synchronous generation in thread pool with timeout
        loop = asyncio.get_running_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._generate_sync, query),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            result.error = f"HyDE generation timed out after {self.config.timeout_seconds}s"
            logger.warning(result.error)
        except Exception as e:
            result.error = str(e)
            logger.error(f"HyDE async expansion failed: {e}")

        return result

    def expand_sync(self, query: str) -> HyDEResult:
        """Synchronously generate a hypothetical document for the query.

        Use this method when not in an async context.

        Args:
            query: The user's search query

        Returns:
            HyDEResult with the hypothetical document or error/skip info
        """
        return self._generate_sync(query)

    def get_query_for_embedding(self, result: HyDEResult) -> str:
        """Get the text to use for embedding based on HyDE result.

        If HyDE expansion succeeded, returns the hypothetical document.
        Otherwise, returns the original query.

        Args:
            result: HyDEResult from expand() or expand_sync()

        Returns:
            Text to use for embedding
        """
        if result.expanded and result.hypothetical_document:
            return result.hypothetical_document
        return result.original_query

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the HyDE expander.

        Useful for health checks and debugging.

        Returns:
            Dictionary with configuration and status information
        """
        return {
            'enabled': self.config.enabled,
            'model': self.config.model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'min_query_length': self.config.min_query_length,
            'max_query_length': self.config.max_query_length,
            'timeout_seconds': self.config.timeout_seconds,
            'skip_patterns': list(self.SKIP_PATTERNS.keys()),
        }


# Singleton instance for use throughout the application
hyde_expander = HyDEExpander()


# Convenience functions for direct usage
async def expand_query(query: str) -> HyDEResult:
    """Convenience function to expand a query using the singleton expander.

    Args:
        query: The user's search query

    Returns:
        HyDEResult with the hypothetical document or error/skip info
    """
    return await hyde_expander.expand(query)


def expand_query_sync(query: str) -> HyDEResult:
    """Convenience function to synchronously expand a query.

    Args:
        query: The user's search query

    Returns:
        HyDEResult with the hypothetical document or error/skip info
    """
    return hyde_expander.expand_sync(query)


def should_use_hyde(query: str) -> bool:
    """Quick check if HyDE should be used for a query.

    Args:
        query: The user's search query

    Returns:
        True if HyDE should be used, False otherwise
    """
    should_expand, _ = hyde_expander.should_expand(query)
    return should_expand
