"""
GeoSense-X — API v1 Router Aggregator
=======================================
Imports and includes all sub-routers under /api/v1.

This file has zero business logic — it only wires routers together.
Add new sub-routers here as each module is implemented.
"""

from __future__ import annotations

from fastapi import APIRouter

# TODO: uncomment each import as the corresponding module is implemented

from api.v1.health import router as health_router
# from api.v1.scenes import router as scenes_router
# from api.v1.retrieve import router as retrieve_router
# from api.v1.explain import router as explain_router
# from api.v1.search import router as search_router
# from api.v1.admin import router as admin_router

api_v1_router = APIRouter(prefix="/api/v1")

# Health and readiness (always included — needed before everything else)
api_v1_router.include_router(health_router, tags=["health"])

# Scene catalog
# api_v1_router.include_router(scenes_router, prefix="/scenes", tags=["scenes"])

# Retrieval endpoints (core feature)
# api_v1_router.include_router(retrieve_router, prefix="/retrieve", tags=["retrieval"])

# Explanation endpoints
# api_v1_router.include_router(explain_router, prefix="/explain", tags=["explanation"])

# Interactive search (WebSocket)
# api_v1_router.include_router(search_router, prefix="/search", tags=["search"])

# Admin endpoints (protected)
# api_v1_router.include_router(admin_router, prefix="/admin", tags=["admin"])
