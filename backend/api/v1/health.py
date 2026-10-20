"""
GeoSense-X — Health & Readiness Endpoints
===========================================
Implements the three observability endpoints:

  GET /health  → liveness probe (always 200 if process is running)
  GET /ready   → readiness probe (200 only if all deps healthy)
  GET /metrics → Prometheus exposition format

These endpoints are used by Kubernetes probes and monitoring dashboards.
They must never be behind authentication middleware.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Response Schemas
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    version: str
    environment: str


class ReadinessCheck(BaseModel):
    """Status of a single dependency check."""
    healthy: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Response body for GET /ready."""
    status: str   # "ready" | "degraded" | "unavailable"
    checks: dict[str, ReadinessCheck]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 if the application process is running. "
                "Used by Kubernetes liveness probe.",
)
async def health() -> HealthResponse:
    """
    Liveness probe endpoint.

    Always returns 200 as long as the process is alive.
    Does not check downstream dependencies.
    """
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 only when all critical dependencies are healthy. "
                "Used by Kubernetes readiness probe.",
)
async def ready(request: Request, response: Response) -> ReadinessResponse:
    """
    Readiness probe endpoint.

    Checks all critical dependencies:
      - PostgreSQL database
      - Redis cache
      - FAISS indexes
      - ML model

    Returns 503 if any critical dependency is unhealthy.
    """
    checks: dict[str, ReadinessCheck] = {}
    all_healthy = True

    # TODO: implement actual health checks once services are loaded
    # Check 1: Database
    # try:
    #     from core.database import health_check as db_health
    #     db_ok = await db_health()
    #     checks["database"] = ReadinessCheck(healthy=db_ok)
    # except Exception as exc:
    #     checks["database"] = ReadinessCheck(healthy=False, detail=str(exc))
    #     all_healthy = False

    # Placeholder checks for scaffold
    checks["database"] = ReadinessCheck(healthy=False, detail="Not yet initialised")
    checks["redis"] = ReadinessCheck(healthy=False, detail="Not yet initialised")
    checks["faiss"] = ReadinessCheck(healthy=False, detail="Not yet initialised")
    checks["model"] = ReadinessCheck(healthy=False, detail="Not yet initialised")
    all_healthy = False

    # Check 2: Redis
    # try:
    #     from core.redis_client import health_check as redis_health
    #     redis_ok = await redis_health()
    #     checks["redis"] = ReadinessCheck(healthy=redis_ok)
    #     if not redis_ok:
    #         all_healthy = False  # Redis is non-fatal — degraded mode allowed
    # except Exception as exc:
    #     checks["redis"] = ReadinessCheck(healthy=False, detail=str(exc))

    # Check 3: FAISS indexes
    # faiss_svc = getattr(request.app.state, "faiss", None)
    # if faiss_svc is None:
    #     checks["faiss"] = ReadinessCheck(healthy=False, detail="Not loaded")
    #     all_healthy = False
    # else:
    #     checks["faiss"] = ReadinessCheck(healthy=True)

    # Check 4: ML model
    # encoder = getattr(request.app.state, "encoder", None)
    # if encoder is None:
    #     checks["model"] = ReadinessCheck(healthy=False, detail="Not loaded")
    #     all_healthy = False
    # else:
    #     checks["model"] = ReadinessCheck(healthy=True)

    if not all_healthy:
        response.status_code = 503

    return ReadinessResponse(
        status="ready" if all_healthy else "unavailable",
        checks=checks,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus exposition format metrics endpoint.",
    response_class=Response,
    include_in_schema=False,
)
async def metrics() -> Response:
    """
    Expose Prometheus metrics in text/plain exposition format.

    All counters, histograms, and gauges defined in core/metrics.py
    are included here automatically via the default registry.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
