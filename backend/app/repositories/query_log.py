"""Repository for query log operations.

Provides data access for query logging and analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import AsyncSessionRepository, RepositoryContext, QueryError, NotFoundError

logger = logging.getLogger(__name__)


class QueryLogRepository(AsyncSessionRepository):
    """Repository for query log operations.

    Provides CRUD operations for query logs stored in PostgreSQL.
    """

    def __init__(
        self,
        session_factory=None,
        context: Optional[RepositoryContext] = None
    ):
        super().__init__(session_factory, context)
        self._model = None

    def _get_model(self):
        """Lazy load model to avoid circular imports."""
        if self._model is None:
            from app.db_models import QueryLog
            self._model = QueryLog
        return self._model

    async def create(self, query_log) -> Any:
        """Create a new query log entry.

        Args:
            query_log: QueryLog instance to create

        Returns:
            Created QueryLog instance
        """
        self._log_operation("create", session_id=query_log.session_id)

        try:
            self._session.add(query_log)
            await self._session.flush()
            return query_log
        except Exception as e:
            self._log_error("create", e)
            raise QueryError(str(e), "create")

    async def find_by_id(self, log_id: int) -> Optional[Any]:
        """Find a query log by ID.

        Args:
            log_id: Query log ID

        Returns:
            QueryLog instance or None
        """
        self._log_operation("find_by_id", log_id=log_id)

        try:
            QueryLog = self._get_model()
            result = await self._session.execute(
                select(QueryLog).where(QueryLog.id == log_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self._log_error("find_by_id", e)
            raise QueryError(str(e), "find_by_id")

    async def find_by_session(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Any]:
        """Find query logs for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of results

        Returns:
            List of QueryLog instances
        """
        self._log_operation("find_by_session", session_id=session_id, limit=limit)

        try:
            QueryLog = self._get_model()
            result = await self._session.execute(
                select(QueryLog)
                .where(QueryLog.session_id == session_id)
                .order_by(desc(QueryLog.created_at))
                .limit(limit)
            )
            return list(result.scalars().all())
        except Exception as e:
            self._log_error("find_by_session", e)
            raise QueryError(str(e), "find_by_session")

    async def get_analytics_summary(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get analytics summary for the specified period.

        Args:
            days: Number of days to include

        Returns:
            Dictionary with analytics summary
        """
        self._log_operation("get_analytics_summary", days=days)

        try:
            QueryLog = self._get_model()
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Total queries
            total_result = await self._session.execute(
                select(func.count(QueryLog.id))
                .where(QueryLog.created_at >= cutoff)
            )
            total_queries = total_result.scalar() or 0

            # Unique sessions
            sessions_result = await self._session.execute(
                select(func.count(func.distinct(QueryLog.session_id)))
                .where(QueryLog.created_at >= cutoff)
            )
            unique_sessions = sessions_result.scalar() or 0

            # Average latency
            latency_result = await self._session.execute(
                select(func.avg(QueryLog.response_time_ms))
                .where(QueryLog.created_at >= cutoff)
            )
            avg_latency = latency_result.scalar() or 0.0

            return {
                "period_days": days,
                "total_queries": total_queries,
                "unique_sessions": unique_sessions,
                "avg_latency_ms": float(avg_latency),
            }
        except Exception as e:
            self._log_error("get_analytics_summary", e)
            raise QueryError(str(e), "get_analytics_summary")

    async def delete_old_logs(self, days: int = 90) -> int:
        """Delete query logs older than specified days.

        Args:
            days: Delete logs older than this many days

        Returns:
            Number of deleted logs
        """
        self._log_operation("delete_old_logs", days=days)

        try:
            from sqlalchemy import delete
            QueryLog = self._get_model()
            cutoff = datetime.utcnow() - timedelta(days=days)

            result = await self._session.execute(
                delete(QueryLog).where(QueryLog.created_at < cutoff)
            )
            return result.rowcount
        except Exception as e:
            self._log_error("delete_old_logs", e)
            raise QueryError(str(e), "delete_old_logs")


__all__ = ["QueryLogRepository"]
