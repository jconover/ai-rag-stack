"""Shared Redis connection pool for the application.

This module provides a centralized Redis connection pool that can be shared
across all components (conversation memory, embedding cache, etc.) to:
- Reduce total connection count to Redis
- Simplify connection management and configuration
- Enable consistent monitoring of Redis connections
- Provide graceful shutdown handling

The pool is configured with decode_responses=False (bytes mode) to support
both string operations (conversation memory) and binary operations (embedding cache).
Callers that need string responses should decode manually or use the string client.

Usage:
    from app.redis_client import get_redis_pool, get_redis_client, close_redis_pool

    # For byte operations (embedding cache)
    client = get_redis_client()
    data = client.get("key")  # Returns bytes

    # For string operations (conversation memory)
    client = get_redis_string_client()
    data = client.get("key")  # Returns str

    # For async code (FastAPI endpoints)
    client = await get_async_redis_client()
"""
import logging
import threading
from typing import Optional

import redis
import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool as AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

# Synchronous connection pools
# We maintain two pools: one for bytes (embedding cache) and one for strings (conversation)
_bytes_redis_pool: Optional[redis.ConnectionPool] = None
_string_redis_pool: Optional[redis.ConnectionPool] = None
_pool_lock = threading.Lock()

# Async connection pool (for FastAPI endpoints)
_async_redis_pool: Optional[AsyncConnectionPool] = None
_async_pool_lock = threading.Lock()


def _create_pool(decode_responses: bool) -> redis.ConnectionPool:
    """Create a Redis connection pool with specified decode_responses setting.

    Args:
        decode_responses: Whether to decode byte responses to strings.

    Returns:
        redis.ConnectionPool: A new connection pool
    """
    return redis.ConnectionPool(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        max_connections=settings.redis_max_connections,
        socket_timeout=settings.redis_socket_timeout,
        socket_connect_timeout=settings.redis_socket_connect_timeout,
        decode_responses=decode_responses,
    )


def get_redis_pool() -> redis.ConnectionPool:
    """Get or create the shared Redis connection pool for byte operations (thread-safe).

    This pool is used for operations that need raw bytes, like the embedding cache.
    For string operations, use get_redis_string_pool() instead.

    Returns:
        redis.ConnectionPool: The shared connection pool (decode_responses=False)
    """
    global _bytes_redis_pool
    if _bytes_redis_pool is None:
        with _pool_lock:
            if _bytes_redis_pool is None:
                _bytes_redis_pool = _create_pool(decode_responses=False)
                logger.info(
                    f"Created shared Redis bytes pool "
                    f"(host={settings.redis_host}, port={settings.redis_port}, "
                    f"max_connections={settings.redis_max_connections})"
                )
    return _bytes_redis_pool


def get_redis_string_pool() -> redis.ConnectionPool:
    """Get or create the shared Redis connection pool for string operations (thread-safe).

    This pool is used for operations that return strings, like conversation history.
    For byte operations, use get_redis_pool() instead.

    Returns:
        redis.ConnectionPool: The shared connection pool (decode_responses=True)
    """
    global _string_redis_pool
    if _string_redis_pool is None:
        with _pool_lock:
            if _string_redis_pool is None:
                _string_redis_pool = _create_pool(decode_responses=True)
                logger.info(
                    f"Created shared Redis string pool "
                    f"(host={settings.redis_host}, port={settings.redis_port}, "
                    f"max_connections={settings.redis_max_connections})"
                )
    return _string_redis_pool


def get_redis_client() -> redis.Redis:
    """Get a synchronous Redis client for byte operations.

    This client returns bytes and is suitable for embedding cache.
    For string operations, use get_redis_string_client() instead.

    Returns:
        redis.Redis: A Redis client instance using the shared bytes pool
    """
    pool = get_redis_pool()
    return redis.Redis(connection_pool=pool)


def get_redis_string_client() -> redis.Redis:
    """Get a synchronous Redis client for string operations.

    This client returns decoded strings and is suitable for conversation history.
    For byte operations, use get_redis_client() instead.

    Returns:
        redis.Redis: A Redis client instance using the shared string pool
    """
    pool = get_redis_string_pool()
    return redis.Redis(connection_pool=pool)


def get_async_redis_pool() -> AsyncConnectionPool:
    """Get or create the shared async Redis connection pool (thread-safe).

    This pool is used for async Redis operations in FastAPI endpoints.

    Returns:
        AsyncConnectionPool: The shared async connection pool
    """
    global _async_redis_pool
    if _async_redis_pool is None:
        with _async_pool_lock:
            if _async_redis_pool is None:
                _async_redis_pool = AsyncConnectionPool(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    max_connections=settings.redis_max_connections,
                    socket_timeout=settings.redis_socket_timeout,
                    socket_connect_timeout=settings.redis_socket_connect_timeout,
                )
                logger.info(
                    f"Created shared async Redis connection pool "
                    f"(host={settings.redis_host}, port={settings.redis_port}, "
                    f"max_connections={settings.redis_max_connections})"
                )
    return _async_redis_pool


async def get_async_redis_client() -> aioredis.Redis:
    """Get an async Redis client using the shared pool.

    Returns:
        aioredis.Redis: An async Redis client instance using the shared pool
    """
    pool = get_async_redis_pool()
    return aioredis.Redis(connection_pool=pool)


def get_redis_pool_stats() -> dict:
    """Get Redis connection pool statistics for monitoring.

    Returns statistics for all pools (bytes, string, async) if they exist.

    Returns:
        Dictionary with pool configuration and current usage stats
    """
    stats = {
        "host": settings.redis_host,
        "port": settings.redis_port,
        "db": settings.redis_db,
        "max_connections": settings.redis_max_connections,
    }

    # Bytes pool stats (for embedding cache)
    if _bytes_redis_pool is not None:
        try:
            stats["current_connections"] = len(_bytes_redis_pool._in_use_connections)
            stats["available_connections"] = len(_bytes_redis_pool._available_connections)
            stats["bytes_pool"] = {
                "current_connections": len(_bytes_redis_pool._in_use_connections),
                "available_connections": len(_bytes_redis_pool._available_connections),
            }
        except Exception as e:
            stats["bytes_pool"] = {"error": str(e)}
    else:
        stats["bytes_pool"] = {"status": "not_initialized"}

    # String pool stats (for conversation memory)
    if _string_redis_pool is not None:
        try:
            # Add to totals
            current = stats.get("current_connections", 0)
            available = stats.get("available_connections", 0)
            stats["current_connections"] = current + len(_string_redis_pool._in_use_connections)
            stats["available_connections"] = available + len(_string_redis_pool._available_connections)
            stats["string_pool"] = {
                "current_connections": len(_string_redis_pool._in_use_connections),
                "available_connections": len(_string_redis_pool._available_connections),
            }
        except Exception as e:
            stats["string_pool"] = {"error": str(e)}
    else:
        stats["string_pool"] = {"status": "not_initialized"}

    # Async pool stats
    if _async_redis_pool is not None:
        stats["async_pool"] = {"status": "initialized"}
    else:
        stats["async_pool"] = {"status": "not_initialized"}

    return stats


def close_redis_pool() -> None:
    """Close all shared Redis connection pools.

    Should be called during application shutdown to cleanly release
    all Redis connections.
    """
    global _bytes_redis_pool, _string_redis_pool

    # Close bytes pool
    if _bytes_redis_pool is not None:
        try:
            _bytes_redis_pool.disconnect()
            logger.info("Closed shared Redis bytes pool")
        except Exception as e:
            logger.warning(f"Error closing Redis bytes pool: {e}")
        finally:
            _bytes_redis_pool = None

    # Close string pool
    if _string_redis_pool is not None:
        try:
            _string_redis_pool.disconnect()
            logger.info("Closed shared Redis string pool")
        except Exception as e:
            logger.warning(f"Error closing Redis string pool: {e}")
        finally:
            _string_redis_pool = None


async def close_async_redis_pool() -> None:
    """Close the shared async Redis connection pool.

    Should be called during application shutdown to cleanly release
    all Redis connections. This is an async function for use in
    FastAPI lifespan handlers.
    """
    global _async_redis_pool
    if _async_redis_pool is not None:
        try:
            await _async_redis_pool.disconnect()
            logger.info("Closed shared async Redis connection pool")
        except Exception as e:
            logger.warning(f"Error closing async Redis pool: {e}")
        finally:
            _async_redis_pool = None


def is_redis_connected() -> bool:
    """Check if Redis is reachable using the shared pool.

    Returns:
        True if Redis ping succeeds, False otherwise
    """
    try:
        client = get_redis_string_client()
        client.ping()
        return True
    except Exception:
        return False
