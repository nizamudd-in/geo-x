"""
GeoSense-X — FastAPI Shared Dependencies
==========================================
Dependency functions injected into route handlers via Depends().

Rules:
  - All shared resource access (DB, Redis, services) goes through dependencies.
  - Never instantiate services directly inside route handlers.
  - Dependencies must be async generators (yield) for proper cleanup.

Usage:
    from api.dependencies import get_db, get_encoder, get_faiss
    async def my_route(db: AsyncSession = Depends(get_db)):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db as _get_db
from core.security import get_current_user

if TYPE_CHECKING:
    from services.encoder_service import EncoderService
    from services.faiss_service import FAISSService
    from services.cache_service import CacheService
    from redis.asyncio import Redis


# --------------------------------------------------------------------------- #
# Database Session
# --------------------------------------------------------------------------- #

async def get_db(request: Request) -> AsyncSession:
    """
    Yield an async database session for the current request.

    Commits on success, rolls back on exception, closes after response.
    Delegates to core/database.py get_db() generator.

    Args:
        request: FastAPI request (unused, but keeps signature uniform).

    Yields:
        AsyncSession: Database session.
    """
    # TODO: implement once core/database.py is complete
    async for session in _get_db():
        yield session


# --------------------------------------------------------------------------- #
# Redis Client
# --------------------------------------------------------------------------- #

async def get_redis(request: Request) -> "Redis | None":
    """
    Return the Redis client from app state.

    Returns None if Redis was unavailable at startup (cache disabled mode).

    Args:
        request: FastAPI request with app.state.redis attached at startup.

    Returns:
        Redis client or None.
    """
    # TODO: return app.state.redis once events.py startup is implemented
    return getattr(request.app.state, "redis", None)


# --------------------------------------------------------------------------- #
# ML Services
# --------------------------------------------------------------------------- #

async def get_encoder(request: Request) -> "EncoderService":
    """
    Return the EncoderService singleton from app state.

    Raises:
        ModelNotLoadedError: If the model has not been loaded yet.
    """
    from core.exceptions import ModelNotLoadedError

    encoder = getattr(request.app.state, "encoder", None)
    if encoder is None:
        raise ModelNotLoadedError()
    return encoder


async def get_faiss(request: Request) -> "FAISSService":
    """
    Return the FAISSService singleton from app state.

    Raises:
        IndexNotLoadedError: If FAISS indexes have not been loaded yet.
    """
    from core.exceptions import IndexNotLoadedError

    faiss = getattr(request.app.state, "faiss", None)
    if faiss is None:
        raise IndexNotLoadedError()
    return faiss


async def get_cache(request: Request) -> "CacheService":
    """
    Return the CacheService singleton from app state.

    The CacheService handles both Redis (hot) and DB (cold) cache tiers.

    Returns:
        CacheService instance.
    """
    # TODO: return app.state.cache once cache_service.py is implemented
    # return request.app.state.cache
    raise NotImplementedError("CacheService not yet initialised")


# --------------------------------------------------------------------------- #
# Rate Limiting
# --------------------------------------------------------------------------- #

async def apply_rate_limit(
    request: Request,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    redis: Annotated["Redis | None", Depends(get_redis)],
) -> None:
    """
    Apply rate limiting for the current request.

    Uses client IP for anonymous users, API key prefix for authenticated ones.
    Does nothing if Redis is unavailable.

    Args:
        request: FastAPI request (for client IP extraction).
        user: Current user dict with tier information.
        redis: Redis client (may be None).
    """
    from core.rate_limiter import RateLimiter

    client_ip = request.client.host if request.client else "unknown"
    client_id = (
        f"key:{user['api_key_prefix']}"
        if user.get("api_key_prefix")
        else f"ip:{client_ip}"
    )
    tier = user.get("tier", "anonymous")

    limiter = RateLimiter(redis)
    await limiter.check(client_id=client_id, tier=tier)


# --------------------------------------------------------------------------- #
# Type Aliases for cleaner route signatures
# --------------------------------------------------------------------------- #

DBSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]
RateLimited = Annotated[None, Depends(apply_rate_limit)]
