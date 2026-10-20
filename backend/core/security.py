"""
GeoSense-X — API Key Authentication & Authorisation
=====================================================
Validates API keys from the Authorization: Bearer <key> header.

For the Phase 1 MVP, a single SHA-256 hashed API key is stored in
config.API_KEY_HASH. Production will migrate to DB-backed key management
with per-key rate limits and tier assignments.

Usage:
    # In a route that requires authentication:
    from core.security import get_current_user
    async def my_route(user: dict = Depends(get_current_user)):
        ...

    # In an admin-only route:
    from core.security import require_admin
    async def admin_route(user: dict = Depends(require_admin)):
        ...
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings
from core.exceptions import AuthenticationError, ForbiddenError
from core.logging import get_logger

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def _hash_key(raw_key: str) -> str:
    """
    Compute SHA-256 hash of a raw API key string.

    Args:
        raw_key: The plaintext API key to hash.

    Returns:
        Lowercase hex digest string.
    """
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> dict[str, Any]:
    """
    FastAPI dependency that validates the Bearer token and returns user context.

    Anonymous requests (no Authorization header) are allowed for public endpoints.
    Returns a minimal user dict with tier="anonymous" for unauthenticated requests.

    Args:
        credentials: Parsed Authorization header or None.

    Returns:
        User dict with keys: user_id, tier, api_key_prefix.

    Raises:
        AuthenticationError: If a Bearer token is provided but invalid.
    """
    if credentials is None:
        # Anonymous access — public endpoints allow this
        return {"user_id": None, "tier": "anonymous", "api_key_prefix": None}

    raw_key = credentials.credentials
    key_hash = _hash_key(raw_key)

    # TODO: replace single-key check with DB lookup in production
    if settings.API_KEY_HASH and key_hash == settings.API_KEY_HASH:
        return {
            "user_id": "api_key_user",
            "tier": "admin",   # MVP: single key is always admin
            "api_key_prefix": raw_key[:8] + "...",
        }

    if settings.API_KEY_HASH:
        # Key was provided but did not match
        logger.warning("Invalid API key attempt", prefix=raw_key[:4] + "***")
        raise AuthenticationError("Invalid API key")

    # No API_KEY_HASH configured — accept any key as authenticated (dev mode)
    # TODO: remove this fallback before production deployment
    logger.debug("API_KEY_HASH not configured — accepting all keys in dev mode")
    return {"user_id": "dev_user", "tier": "api_key", "api_key_prefix": None}


async def require_admin(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    FastAPI dependency that requires admin tier access.

    Use as a dependency on admin-only endpoints.

    Args:
        user: User dict from get_current_user.

    Returns:
        Same user dict if authorised.

    Raises:
        AuthenticationError: If no credentials were provided.
        ForbiddenError: If user tier is not "admin".
    """
    if user["user_id"] is None:
        raise AuthenticationError("Admin endpoints require authentication")
    if user["tier"] != "admin":
        raise ForbiddenError("Admin tier required for this endpoint")
    return user
