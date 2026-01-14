"""Base repository pattern implementation with async context management.

This module provides the abstract base class for all repositories in the
DevOps AI Assistant, establishing a consistent interface for data access
with proper async context management and dependency injection support.

Usage:
    class MyRepository(BaseRepository[MyEntity]):
        async def find_by_id(self, id: str) -> Optional[MyEntity]:
            ...
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Any, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Generic type for entity types
T = TypeVar("T")


class RepositoryError(Exception):
    """Base exception for repository operations."""

    def __init__(self, message: str, operation: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.details = details or {}

    def __str__(self) -> str:
        if self.operation:
            return f"[{self.operation}] {self.message}"
        return self.message


class ConnectionError(RepositoryError):
    """Raised when a connection to the data store cannot be established."""
    pass


class QueryError(RepositoryError):
    """Raised when a query fails to execute."""
    pass


class NotFoundError(RepositoryError):
    """Raised when a requested entity is not found."""
    pass


class ValidationError(RepositoryError):
    """Raised when entity validation fails."""
    pass


class DuplicateError(RepositoryError):
    """Raised when attempting to create a duplicate entity."""
    pass


@dataclass
class RepositoryContext:
    """Context information for repository operations.

    Provides metadata about the current operation context,
    useful for logging, tracing, and audit purposes.
    """
    operation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for repositories with async context management.

    Provides a consistent interface for data access operations with:
    - Async context management for resource cleanup
    - Generic type hints for entity types
    - Error handling patterns
    - Logging integration
    - Dependency injection support

    All repositories should inherit from this class and implement
    the required abstract methods for their specific data store.

    Type Parameters:
        T: The entity type this repository manages

    Example:
        class UserRepository(BaseRepository[User]):
            async def find_by_id(self, user_id: str) -> Optional[User]:
                ...

            async def save(self, user: User) -> User:
                ...
    """

    def __init__(self, context: Optional[RepositoryContext] = None):
        """Initialize repository with optional context.

        Args:
            context: Optional context information for tracing and audit
        """
        self._context = context or RepositoryContext()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_connected = False

    @property
    def context(self) -> RepositoryContext:
        """Get the current repository context."""
        return self._context

    @context.setter
    def context(self, value: RepositoryContext) -> None:
        """Set the repository context."""
        self._context = value

    @property
    def is_connected(self) -> bool:
        """Check if the repository is connected to its data store."""
        return self._is_connected

    async def __aenter__(self) -> "BaseRepository[T]":
        """Async context manager entry - establish connection."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        await self.disconnect()
        # Don't suppress exceptions
        return None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data store.

        Should be called before performing any operations.
        May be called automatically via async context manager.

        Raises:
            ConnectionError: If connection cannot be established
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection and cleanup resources.

        Should be called after all operations are complete.
        May be called automatically via async context manager.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the data store is healthy and responsive.

        Returns:
            True if healthy, False otherwise
        """
        pass

    def _log_operation(self, operation: str, **kwargs) -> None:
        """Log repository operation with context."""
        extra = {
            "operation": operation,
            "repository": self.__class__.__name__,
            **kwargs,
        }
        if self._context.operation_id:
            extra["operation_id"] = self._context.operation_id
        if self._context.trace_id:
            extra["trace_id"] = self._context.trace_id

        self._logger.debug(f"Repository operation: {operation}", extra=extra)

    def _log_error(self, operation: str, error: Exception, **kwargs) -> None:
        """Log repository error with context."""
        extra = {
            "operation": operation,
            "repository": self.__class__.__name__,
            "error_type": type(error).__name__,
            **kwargs,
        }
        if self._context.operation_id:
            extra["operation_id"] = self._context.operation_id

        self._logger.error(
            f"Repository error in {operation}: {error}",
            extra=extra,
            exc_info=True,
        )


class AsyncSessionRepository(BaseRepository[T]):
    """Base repository for SQLAlchemy async session-based data access.

    Extends BaseRepository with SQLAlchemy-specific functionality
    for working with async sessions and the PostgreSQL database.

    This class handles session lifecycle management and provides
    helper methods for common database operations.
    """

    def __init__(
        self,
        session_factory=None,
        context: Optional[RepositoryContext] = None,
    ):
        """Initialize with optional session factory.

        Args:
            session_factory: SQLAlchemy async session factory
            context: Optional context information
        """
        super().__init__(context)
        self._session_factory = session_factory
        self._session = None

    async def connect(self) -> None:
        """Establish database session."""
        if self._session is not None:
            return

        if self._session_factory is None:
            # Import here to avoid circular imports
            from app.database import get_session_factory
            self._session_factory = get_session_factory()

        self._session = self._session_factory()
        self._is_connected = True
        self._log_operation("connect")

    async def disconnect(self) -> None:
        """Close database session."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._is_connected = False
        self._log_operation("disconnect")

    async def health_check(self) -> bool:
        """Check database connection health."""
        try:
            from sqlalchemy import text
            await self._session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self._log_error("health_check", e)
            return False

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions.

        Usage:
            async with repo.transaction():
                await repo.save(entity1)
                await repo.save(entity2)
                # Commits if no exceptions, rolls back otherwise
        """
        try:
            yield self._session
            await self._session.commit()
        except Exception as e:
            await self._session.rollback()
            self._log_error("transaction", e)
            raise
