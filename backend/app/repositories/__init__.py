"""Repository pattern implementations for data access.

This package provides repository classes that abstract data access
for different storage systems (Qdrant, PostgreSQL, Redis).

Usage:
    from app.repositories import VectorRepository, QueryLogRepository, ConversationRepository

    async with VectorRepository() as repo:
        results = await repo.search_with_scores("kubernetes pods", top_k=5)
"""

from app.repositories.base import (
    BaseRepository,
    AsyncSessionRepository,
    RepositoryContext,
    RepositoryError,
    ConnectionError,
    QueryError,
    NotFoundError,
    ValidationError,
    DuplicateError,
)
from app.repositories.vector import VectorRepository
from app.repositories.query_log import QueryLogRepository
from app.repositories.conversation import ConversationRepository

__all__ = [
    # Base classes
    "BaseRepository",
    "AsyncSessionRepository",
    "RepositoryContext",
    # Exceptions
    "RepositoryError",
    "ConnectionError",
    "QueryError",
    "NotFoundError",
    "ValidationError",
    "DuplicateError",
    # Concrete repositories
    "VectorRepository",
    "QueryLogRepository",
    "ConversationRepository",
]
