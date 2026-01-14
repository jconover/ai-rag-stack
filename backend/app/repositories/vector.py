"""Repository for vector search operations.

Wraps the VectorStore with repository pattern for clean data access.
"""

import logging
from typing import Optional, List, Tuple, Dict, Any

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from app.repositories.base import BaseRepository, RepositoryContext, QueryError

logger = logging.getLogger(__name__)


class VectorRepository(BaseRepository[Document]):
    """Repository for vector search operations.

    Provides a clean interface for vector store operations while
    wrapping the existing VectorStore implementation.
    """

    def __init__(self, context: Optional[RepositoryContext] = None):
        super().__init__(context)
        self._vector_store = None

    def _get_store(self):
        """Lazy load vector store to avoid circular imports."""
        if self._vector_store is None:
            from app.vectorstore import vector_store
            self._vector_store = vector_store
        return self._vector_store

    async def connect(self) -> None:
        """Verify vector store connection."""
        try:
            store = self._get_store()
            if store is None:
                raise QueryError("Vector store not initialized", "connect")
            self._is_connected = True
            self._log_operation("connect")
        except Exception as e:
            self._log_error("connect", e)
            raise

    async def disconnect(self) -> None:
        """Cleanup (no-op for vector store singleton)."""
        self._is_connected = False
        self._log_operation("disconnect")

    async def health_check(self) -> bool:
        """Check vector store health."""
        try:
            store = self._get_store()
            stats = store.get_collection_stats()
            return stats is not None
        except Exception as e:
            self._log_error("health_check", e)
            return False

    async def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using vector similarity.

        Args:
            query: Search query string
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold

        Returns:
            List of (Document, score) tuples
        """
        self._log_operation("search_with_scores", query_length=len(query), top_k=top_k)

        try:
            store = self._get_store()
            results, _ = store.search_with_cache_info(
                query=query,
                top_k=top_k,
                min_score=min_score
            )
            return results
        except Exception as e:
            self._log_error("search_with_scores", e)
            raise QueryError(str(e), "search_with_scores")

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Search using hybrid (dense + sparse) retrieval.

        Args:
            query: Search query string
            top_k: Maximum number of results
            min_score: Minimum score threshold

        Returns:
            List of (Document, score) tuples
        """
        self._log_operation("hybrid_search", query_length=len(query), top_k=top_k)

        try:
            store = self._get_store()
            results, _ = store.hybrid_search_with_cache_info(
                query=query,
                top_k=top_k,
                min_score=min_score
            )
            return results
        except Exception as e:
            self._log_error("hybrid_search", e)
            raise QueryError(str(e), "hybrid_search")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector collection statistics.

        Returns:
            Dictionary with collection statistics
        """
        self._log_operation("get_collection_stats")

        try:
            store = self._get_store()
            return store.get_collection_stats()
        except Exception as e:
            self._log_error("get_collection_stats", e)
            raise QueryError(str(e), "get_collection_stats")

    async def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add
            batch_size: Batch size for insertion

        Returns:
            Number of documents added
        """
        self._log_operation("add_documents", count=len(documents))

        try:
            store = self._get_store()
            store.add_documents(documents, batch_size=batch_size)
            return len(documents)
        except Exception as e:
            self._log_error("add_documents", e)
            raise QueryError(str(e), "add_documents")


__all__ = ["VectorRepository"]
