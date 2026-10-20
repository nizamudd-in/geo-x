"""
GeoSense-X — Sliding Window Rate Limiter
==========================================
Implements per-client rate limiting using Redis sorted sets.

Algorithm: sliding window log
  - Add current timestamp to a sorted set keyed by client identifier.
  - Remove timestamps older than the window (60 seconds).
  - Count remaining entries; reject if count > limit.

Tiers:
  anonymous → 10 req/min
  api_key   → 60 req/min
  premium   → 600 req/min

Degrades gracefully when Redis is unavailable — rate limiting is
skipped with a warning rather than blocking all requests.

Usage:
    from core.rate_limiter import RateLimiter
    limiter = RateLimiter(redis_client)
    await limiter.check(client_id="ip:1.2.3.4", tier="anonymous")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from redis.asyncio import Redis

from config import settings
from core.exceptions import RateLimitExceeded
from core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Rate limit tiers — requests per window
_LIMITS: dict[str, int] = {
    "anonymous": settings.RATE_LIMIT_ANONYMOUS_PER_MINUTE,
    "api_key": settings.RATE_LIMIT_API_KEY_PER_MINUTE,
    "premium": settings.RATE_LIMIT_PREMIUM_PER_MINUTE,
    "admin": settings.RATE_LIMIT_PREMIUM_PER_MINUTE,
}

_WINDOW_SECONDS: int = 60


class RateLimiter:
    """
    Sliding window rate limiter backed by Redis sorted sets.

    Each client has a sorted set where members are request timestamps
    and scores are the same timestamps (for range-based eviction).
    """

    def __init__(self, redis: Redis | None) -> None:
        """
        Initialise the rate limiter.

        Args:
            redis: Connected Redis client, or None to disable rate limiting.
        """
        self._redis = redis

    async def check(self, client_id: str, tier: str = "anonymous") -> None:
        """
        Check whether the client is within their rate limit.

        Increments the request counter for the client.
        Raises RateLimitExceeded if the limit is breached.

        Args:
            client_id: Unique identifier for the client (e.g., "ip:1.2.3.4"
                       or "key:abc123").
            tier: Rate limit tier name.

        Raises:
            RateLimitExceeded: If the client has exceeded their quota.
        """
        if self._redis is None:
            # Redis unavailable — skip rate limiting
            logger.debug("Rate limiting skipped (Redis unavailable)")
            return

        limit = _LIMITS.get(tier, _LIMITS["anonymous"])
        now = time.time()
        window_start = now - _WINDOW_SECONDS
        key = f"rl:{client_id}"

        # TODO: wrap in a Lua script for atomicity in production
        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, "-inf", window_start)  # evict old entries
            pipe.zadd(key, {str(now): now})                  # add current request
            pipe.zcard(key)                                   # count requests in window
            pipe.expire(key, _WINDOW_SECONDS * 2)            # TTL cleanup
            results = await pipe.execute()

            request_count = results[2]

            if request_count > limit:
                logger.warning(
                    "Rate limit exceeded",
                    client_id=client_id,
                    tier=tier,
                    count=request_count,
                    limit=limit,
                )
                raise RateLimitExceeded(limit=limit, window_seconds=_WINDOW_SECONDS)

        except RateLimitExceeded:
            raise
        except Exception as exc:
            # Non-rate-limit Redis error — degrade gracefully
            logger.warning("Rate limiter error, skipping check", error=str(exc))
