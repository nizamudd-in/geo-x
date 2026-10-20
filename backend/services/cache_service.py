"""
GeoSense-X — Two-Level Cache Service
======================================
Abstracts the two-level cache (Redis hot + PostgreSQL cold) into a
single interface used by explanation and confidence services.

Key patterns (Redis):
  embed:{sha256(image)}          → embedding bytes          TTL: 24h
  expl:{query_id}:{scene_id}     → ExplanationBundle JSON   TTL: 7d
  conf:{query_id}:{scene_id}     → ConfidenceBundle JSON    TTL: 7d
  heatmap:{qid}:{sid}:{view}     → PNG bytes                TTL: 7d
  scene_expl:{scene_id}          → SceneExplanation JSON    TTL: 30d
  calib:{model_version}          → CalibrationTable JSON    TTL: 1h
  axis:{model_version}           → AxisVectors JSON         TTL: 1h
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from config import settings
from core.logging import get_logger
from core.metrics import CACHE_HIT_TOTAL, CACHE_MISS_TOTAL

logger = get_logger(__name__)


class CacheService:
    """
    Unified cache interface over Redis (hot) and PostgreSQL (cold).

    Instantiated once at startup and stored in app.state.cache.
    Falls back gracefully when Redis is unavailable.
    """

    def __init__(self, redis: Any | None) -> None:
        """
        Initialise the cache service.

        Args:
            redis: aioredis Redis client, or None to disable hot cache.
        """
        self._redis = redis

    # ---------------------------------------------------------------------- #
    # Raw byte operations
    # ---------------------------------------------------------------------- #

    async def get(self, key: str) -> bytes | None:
        """
        Get raw bytes from Redis.

        Args:
            key: Cache key string.

        Returns:
            Cached bytes or None on miss/unavailable.
        """
        if self._redis is None:
            return None
        try:
            value = await self._redis.get(key)
            if value is not None:
                CACHE_HIT_TOTAL.labels(cache_type=key.split(":")[0]).inc()
            else:
                CACHE_MISS_TOTAL.labels(cache_type=key.split(":")[0]).inc()
            return value
        except Exception as exc:
            logger.debug("Cache get failed", key=key, error=str(exc))
            return None

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """
        Set raw bytes in Redis with optional TTL.

        Args:
            key: Cache key string.
            value: Bytes to cache.
            ttl: Time-to-live in seconds. None means no expiry.
        """
        if self._redis is None:
            return
        try:
            if ttl:
                await self._redis.setex(key, ttl, value)
            else:
                await self._redis.set(key, value)
        except Exception as exc:
            logger.debug("Cache set failed", key=key, error=str(exc))

    # ---------------------------------------------------------------------- #
    # JSON operations
    # ---------------------------------------------------------------------- #

    async def get_json(self, key: str) -> dict | None:
        """Get and deserialise a JSON-encoded cache value."""
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Cache JSON decode failed", key=key)
            return None

    async def set_json(self, key: str, value: dict, ttl: int | None = None) -> None:
        """Serialise a dict to JSON and store in cache."""
        try:
            raw = json.dumps(value, default=str).encode()
            await self.set(key, raw, ttl)
        except Exception as exc:
            logger.debug("Cache set_json failed", key=key, error=str(exc))

    # ---------------------------------------------------------------------- #
    # Embedding operations (binary float32)
    # ---------------------------------------------------------------------- #

    async def get_embedding(self, image_hash: str) -> np.ndarray | None:
        """
        Retrieve a cached embedding vector.

        Args:
            image_hash: SHA-256 hash of the raw image bytes.

        Returns:
            np.ndarray [128] float32 or None on miss.
        """
        key = f"embed:{image_hash}"
        raw = await self.get(key)
        if raw is None:
            return None
        # TODO: use struct.unpack for float32 array
        try:
            arr = np.frombuffer(raw, dtype=np.float32)
            return arr
        except Exception:
            return None

    async def set_embedding(self, image_hash: str, embedding: np.ndarray) -> None:
        """
        Cache an embedding vector as raw float32 bytes.

        Args:
            image_hash: SHA-256 hash of the source image bytes.
            embedding: [128] float32 L2-normalised embedding.
        """
        key = f"embed:{image_hash}"
        raw = embedding.astype(np.float32).tobytes()
        await self.set(key, raw, ttl=settings.EMBEDDING_CACHE_TTL_SECONDS)

    # ---------------------------------------------------------------------- #
    # Explanation/Confidence shortcuts
    # ---------------------------------------------------------------------- #

    async def get_explanation(self, query_id: str, scene_id: str) -> dict | None:
        """Retrieve a cached ExplanationBundle."""
        return await self.get_json(f"expl:{query_id}:{scene_id}")

    async def set_explanation(self, query_id: str, scene_id: str, bundle: dict) -> None:
        """Cache an ExplanationBundle."""
        await self.set_json(
            f"expl:{query_id}:{scene_id}",
            bundle,
            ttl=settings.EXPLANATION_CACHE_TTL_SECONDS,
        )

    async def get_confidence(self, query_id: str, scene_id: str) -> dict | None:
        """Retrieve a cached ConfidenceBundle."""
        return await self.get_json(f"conf:{query_id}:{scene_id}")

    async def set_confidence(self, query_id: str, scene_id: str, bundle: dict) -> None:
        """Cache a ConfidenceBundle."""
        await self.set_json(
            f"conf:{query_id}:{scene_id}",
            bundle,
            ttl=settings.EXPLANATION_CACHE_TTL_SECONDS,
        )

    async def get_scene_explanation(self, scene_id: str) -> dict | None:
        """Retrieve precomputed per-scene explanation signals."""
        return await self.get_json(f"scene_expl:{scene_id}")

    async def set_scene_explanation(self, scene_id: str, signals: dict) -> None:
        """Cache precomputed per-scene explanation signals."""
        await self.set_json(
            f"scene_expl:{scene_id}",
            signals,
            ttl=settings.SCENE_EXPLANATION_CACHE_TTL_SECONDS,
        )

    async def get_heatmap(self, query_id: str, scene_id: str, view: str) -> bytes | None:
        """Retrieve a cached heatmap PNG."""
        key = f"heatmap:{query_id}:{scene_id}:{view[0]}"
        return await self.get(key)

    async def set_heatmap(
        self, query_id: str, scene_id: str, view: str, png_bytes: bytes
    ) -> None:
        """Cache a heatmap PNG."""
        key = f"heatmap:{query_id}:{scene_id}:{view[0]}"
        await self.set(key, png_bytes, ttl=settings.HEATMAP_CACHE_TTL_SECONDS)

    # ---------------------------------------------------------------------- #
    # Admin operations
    # ---------------------------------------------------------------------- #

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all Redis keys matching a glob pattern.

        Args:
            pattern: Redis glob pattern, e.g. "expl:*".

        Returns:
            Number of keys deleted.
        """
        if self._redis is None:
            return 0
        try:
            # TODO: use SCAN instead of KEYS for production safety
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0
        except Exception as exc:
            logger.error("Cache delete_pattern failed", pattern=pattern, error=str(exc))
            return 0
