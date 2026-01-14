"""Repository for conversation history operations.

Wraps Redis-backed conversation storage with repository pattern.
"""

import logging
from typing import Optional, List, Dict, Any

from app.repositories.base import BaseRepository, RepositoryContext, QueryError

logger = logging.getLogger(__name__)


class ConversationRepository(BaseRepository[List[Dict[str, str]]]):
    """Repository for conversation history operations.

    Provides a clean interface for conversation storage while
    wrapping the existing ConversationStorage implementation.
    """

    def __init__(self, context: Optional[RepositoryContext] = None):
        super().__init__(context)
        self._storage = None

    def _get_storage(self):
        """Lazy load storage to avoid circular imports."""
        if self._storage is None:
            from app.conversation_storage import conversation_storage
            self._storage = conversation_storage
        return self._storage

    async def connect(self) -> None:
        """Verify storage connection."""
        try:
            storage = self._get_storage()
            if storage is None:
                raise QueryError("Conversation storage not initialized", "connect")
            self._is_connected = True
            self._log_operation("connect")
        except Exception as e:
            self._log_error("connect", e)
            raise

    async def disconnect(self) -> None:
        """Cleanup (no-op for storage singleton)."""
        self._is_connected = False
        self._log_operation("disconnect")

    async def health_check(self) -> bool:
        """Check storage health."""
        try:
            storage = self._get_storage()
            return storage.is_available()
        except Exception as e:
            self._log_error("health_check", e)
            return False

    async def get_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        self._log_operation("get_history", session_id=session_id, limit=limit)

        try:
            storage = self._get_storage()
            return await storage.get_history_async(session_id, limit=limit)
        except Exception as e:
            self._log_error("get_history", e)
            raise QueryError(str(e), "get_history")

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """Add a message to conversation history.

        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self._log_operation("add_message", session_id=session_id, role=role)

        try:
            storage = self._get_storage()
            await storage.add_message_async(session_id, role, content)
        except Exception as e:
            self._log_error("add_message", e)
            raise QueryError(str(e), "add_message")

    async def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        self._log_operation("clear_history", session_id=session_id)

        try:
            storage = self._get_storage()
            await storage.clear_history_async(session_id)
        except Exception as e:
            self._log_error("clear_history", e)
            raise QueryError(str(e), "clear_history")

    async def get_context(
        self,
        session_id: str,
        max_recent: int = 5
    ) -> str:
        """Get formatted conversation context for RAG.

        Args:
            session_id: Session identifier
            max_recent: Maximum recent messages to include

        Returns:
            Formatted context string
        """
        self._log_operation("get_context", session_id=session_id)

        try:
            storage = self._get_storage()
            return await storage.get_context_async(session_id, max_recent=max_recent)
        except Exception as e:
            self._log_error("get_context", e)
            raise QueryError(str(e), "get_context")

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        self._log_operation("get_session_stats", session_id=session_id)

        try:
            history = await self.get_history(session_id, limit=1000)
            user_messages = sum(1 for m in history if m.get('role') == 'user')
            assistant_messages = sum(1 for m in history if m.get('role') == 'assistant')

            return {
                "session_id": session_id,
                "total_messages": len(history),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
            }
        except Exception as e:
            self._log_error("get_session_stats", e)
            raise QueryError(str(e), "get_session_stats")


__all__ = ["ConversationRepository"]
