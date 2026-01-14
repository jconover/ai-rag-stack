"""Base classes for retrieval strategies.

This module defines the core interfaces and data structures for implementing
retrieval strategies in the RAG pipeline.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result container for retrieval operations.

    Attributes:
        documents: List of retrieved documents
        scores: Similarity/relevance scores for each document
        metadata: Additional metadata about the retrieval operation
        timing_ms: Time taken for the retrieval in milliseconds
        strategy_name: Name of the strategy that produced this result
        cache_hit: Whether the result came from cache
        error: Error message if retrieval failed
    """
    documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing_ms: float = 0.0
    strategy_name: str = ""
    cache_hit: Optional[bool] = None
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        """Check if the result contains no documents."""
        return len(self.documents) == 0

    @property
    def count(self) -> int:
        """Get the number of retrieved documents."""
        return len(self.documents)

    @property
    def avg_score(self) -> float:
        """Calculate average score of retrieved documents."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def max_score(self) -> float:
        """Get the maximum score among retrieved documents."""
        if not self.scores:
            return 0.0
        return max(self.scores)

    @property
    def min_score(self) -> float:
        """Get the minimum score among retrieved documents."""
        if not self.scores:
            return 0.0
        return min(self.scores)

    def to_documents_with_scores(self) -> List[Tuple[Document, float]]:
        """Convert to list of (document, score) tuples."""
        return list(zip(self.documents, self.scores))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "count": self.count,
            "scores": self.scores,
            "avg_score": self.avg_score,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "timing_ms": self.timing_ms,
            "strategy_name": self.strategy_name,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "metadata": self.metadata,
        }


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies.

    All retrieval strategies must implement this interface to be used
    in the RAG pipeline. Strategies can be composed and chained for
    complex retrieval workflows.

    Example:
        class CustomStrategy(RetrievalStrategy):
            def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
                # Custom retrieval logic
                pass

            async def retrieve_async(self, query: str, top_k: int = 5) -> RetrievalResult:
                # Async version
                pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this retrieval strategy."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve documents relevant to the query.

        Args:
            query: The search query string
            top_k: Maximum number of documents to retrieve
            min_score: Minimum score threshold for filtering results
            **kwargs: Additional strategy-specific parameters

        Returns:
            RetrievalResult containing documents, scores, and metadata
        """
        pass

    @abstractmethod
    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        **kwargs
    ) -> RetrievalResult:
        """Async version of retrieve.

        Args:
            query: The search query string
            top_k: Maximum number of documents to retrieve
            min_score: Minimum score threshold for filtering results
            **kwargs: Additional strategy-specific parameters

        Returns:
            RetrievalResult containing documents, scores, and metadata
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration of this strategy.

        Returns:
            Dictionary of configuration parameters
        """
        return {}

    def is_available(self) -> bool:
        """Check if this strategy is available for use.

        Returns:
            True if the strategy can be used, False otherwise
        """
        return True


# Re-export Document for convenience
__all__ = ["RetrievalStrategy", "RetrievalResult", "Document"]
