"""PostgreSQL database connection and session management using async SQLAlchemy.

This module provides async database connectivity for the DevOps AI Assistant,
supporting query logging, feedback storage, and analytics.

Usage:
    from app.database import get_db, init_db, check_postgres_connection

    # Initialize tables on startup
    await init_db()

    # Check connection health
    connected = await check_postgres_connection()

    # Use in endpoints with dependency injection
    @app.get("/endpoint")
    async def endpoint(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Model))
        return result.scalars().all()
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""
    pass


# Global engine and session factory - initialized lazily
_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    """Construct PostgreSQL async connection URL from settings.

    Returns:
        Async PostgreSQL connection string for asyncpg driver
    """
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def get_engine() -> AsyncEngine:
    """Get or create the async database engine.

    Uses connection pooling with configurable pool size for optimal performance.
    Pool settings can be tuned via environment variables.

    Returns:
        AsyncEngine instance
    """
    global _engine

    if _engine is None:
        database_url = get_database_url()

        # Configure connection pool based on settings
        pool_kwargs = {}
        if settings.postgres_pool_size > 0:
            pool_kwargs = {
                "pool_size": settings.postgres_pool_size,
                "max_overflow": settings.postgres_max_overflow,
                "pool_timeout": settings.postgres_pool_timeout,
                "pool_recycle": settings.postgres_pool_recycle,
                "pool_pre_ping": True,  # Verify connections before use
            }
        else:
            # Use NullPool for testing or single-connection scenarios
            pool_kwargs = {"poolclass": NullPool}

        _engine = create_async_engine(
            database_url,
            echo=settings.postgres_echo_sql,
            **pool_kwargs,
        )

        logger.info(
            f"PostgreSQL engine created: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.

    Returns:
        async_sessionmaker configured for the database engine
    """
    global _async_session_factory

    if _async_session_factory is None:
        engine = get_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions.

    Yields:
        AsyncSession that auto-closes after use

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions (for use outside FastAPI dependencies).

    Usage:
        async with get_db_context() as db:
            result = await db.execute(select(Model))
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables.

    Creates all tables defined in db_models.py if they don't exist.
    Safe to call multiple times - uses CREATE TABLE IF NOT EXISTS.
    """
    # Import models to register them with Base.metadata
    from app import db_models  # noqa: F401

    engine = get_engine()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("PostgreSQL database tables initialized")


async def check_postgres_connection() -> bool:
    """Check if PostgreSQL is connected and responsive.

    Executes a simple query to verify the connection is working.

    Returns:
        True if connected, False otherwise
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"PostgreSQL connection check failed: {e}")
        return False


async def get_postgres_pool_stats() -> dict:
    """Get PostgreSQL connection pool statistics.

    Returns:
        Dictionary with pool configuration and current usage stats
    """
    try:
        engine = get_engine()
        pool = engine.pool

        return {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "database": settings.postgres_db,
            "pool_size": settings.postgres_pool_size,
            "max_overflow": settings.postgres_max_overflow,
            "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else None,
            "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else None,
            "overflow": pool.overflow() if hasattr(pool, 'overflow') else None,
            "pool_timeout": settings.postgres_pool_timeout,
        }
    except Exception as e:
        return {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "database": settings.postgres_db,
            "error": str(e),
        }


async def close_db() -> None:
    """Close database connections and dispose of the engine.

    Call this on application shutdown to cleanly close all connections.
    """
    global _engine, _async_session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("PostgreSQL connections closed")
