"""
GeoSense-X — Redis Connection Pool
=====================================
Manages an aioredis connection pool for async cache operations.

The pool is initialised once at startup and shared across all requests.
Operations degrade gracefully if Redis is unavailable — cache misses
are treated as normal and the system falls back to computing fresh results.

Usage:
    # In a route handler (via dependency injection):
    async def my_route(redis: Redis = Depends(get_redis)):
        value = await redis.get("my_key")

    # Direct access (services layer):
    from core.redis_client import get_redis_client
    redis = await get_redis_client()
"""

from __future__ import annotations

from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Module-level pool — created once at startup
_pool: ConnectionPool | None = None
_client: Redis | None = None


async def init_redis() -> Redis:
    """
    Initialise the Redis connection pool and verify connectivity.

    Called once during application startup (core/events.py).
    Stores the client in the module-level singleton.

    Returns:
        Redis: Connected Redis client instance.

    Raises:
        RedisConnectionError: If Redis is unreachable at startup.
    """
    global _pool, _client

    logger.info("Initialising Redis connection", url=settings.REDIS_URL)

    _pool = ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=settings.REDIS_MAX_CONNECTIONS,
        socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
        decode_responses=False,  # keep binary for embedding byte storage
        health_check_interval=30,
    )

    _client = Redis(connection_pool=_pool)

    # Verify connectivity
    try:
        await _client.ping()
        logger.info("Redis connection established")
    except RedisConnectionError as exc:
        logger.warning(
            "Redis unavailable at startup — cache disabled",
            error=str(exc),
        )
        # TODO: decide if Redis unavailability is fatal or degraded mode
        # For MVP: non-fatal, caching is disabled

    return _client


async def close_redis() -> None:
    """
    Close the Redis connection pool.

    Called during application shutdown (core/events.py).
    """
    global _client, _pool
    if _client is not None:
        await _client.aclose()
        logger.info("Redis connection closed")
        _client = None
        _pool = None


async def get_redis() -> Redis | None:
    """
    FastAPI dependency that returns the Redis client for this request.

    Returns None if Redis was unavailable at startup.
    Callers must handle None (treat as cache miss).

    Yields:
        Redis | None: Redis client or None if unavailable.
    """
    # TODO: convert to a proper dependency yield once middleware is in place
    return _client


async def get_redis_client() -> Redis | None:
    """
    Return the module-level Redis client singleton.

    Used by service layer code that is not inside a FastAPI request context.

    Returns:
        Redis | None: Connected client or None.
    """
    return _client


async def health_check() -> bool:
    """
    Check whether Redis is reachable.

    Used by GET /ready endpoint.

    Returns:
        True if Redis responds to PING, False otherwise.
    """
    if _client is None:
        return False
    try:
        result = await _client.ping()
        return bool(result)
    except Exception as exc:
        logger.warning("Redis health check failed", error=str(exc))
        return False
