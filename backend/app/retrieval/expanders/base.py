"""Base classes for query expansion strategies.

This module defines the core interfaces and data structures for implementing
query expanders in the RAG pipeline.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """Result container for query expansion operations.

    Attributes:
        original_query: The original query before expansion
        expanded_query: The expanded/enriched query for retrieval
        expanded: Whether expansion was performed
        skip_reason: Reason why expansion was skipped (if applicable)
        context_terms: List of context terms extracted or used
        metadata: Additional metadata about the expansion
        expansion_time_ms: Time taken for expansion in milliseconds
        expander_name: Name of the expander that produced this result
        error: Error message if expansion failed
    """
    original_query: str
    expanded_query: str
    expanded: bool = False
    skip_reason: Optional[str] = None
    context_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expansion_time_ms: float = 0.0
    expander_name: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses or logging."""
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "expanded": self.expanded,
            "skip_reason": self.skip_reason,
            "context_terms": self.context_terms,
            "expansion_time_ms": self.expansion_time_ms,
            "expander_name": self.expander_name,
            "error": self.error,
            "metadata": self.metadata,
        }


class QueryExpander(ABC):
    """Abstract base class for query expansion strategies.

    Query expanders transform or enrich queries to improve retrieval quality.
    Examples include:
    - HyDE: Generate hypothetical documents for semantic matching
    - Conversation: Resolve pronouns using conversation history
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this expander."""
        pass

    @abstractmethod
    def expand(self, query: str, **kwargs) -> ExpansionResult:
        """Expand the query synchronously."""
        pass

    @abstractmethod
    async def expand_async(self, query: str, **kwargs) -> ExpansionResult:
        """Expand the query asynchronously."""
        pass

    def should_expand(self, query: str, **kwargs) -> Tuple[bool, Optional[str]]:
        """Check if this expander should be applied to the query."""
        return True, None

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of this expander."""
        return {"name": self.name}

    def is_available(self) -> bool:
        """Check if this expander is available for use."""
        return True


class BaseExpander(QueryExpander):
    """Base implementation with common functionality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def expand_async(self, query: str, **kwargs) -> ExpansionResult:
        """Default async implementation runs sync expand in executor."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.expand(query, **kwargs))

    def _create_result(
        self,
        original_query: str,
        expanded_query: str,
        expanded: bool = True,
        **kwargs
    ) -> ExpansionResult:
        """Helper to create ExpansionResult with expander name set."""
        return ExpansionResult(
            original_query=original_query,
            expanded_query=expanded_query,
            expanded=expanded,
            expander_name=self.name,
            **kwargs
        )


__all__ = ["QueryExpander", "BaseExpander", "ExpansionResult"]
