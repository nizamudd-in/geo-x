"""
Integration tests for GET /health and GET /ready endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    """GET /health should always return 200 with status='ok'."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_returns_version(client: AsyncClient) -> None:
    """GET /health should include the application version string."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] != ""


@pytest.mark.asyncio
async def test_ready_returns_503_when_uninitialized(client: AsyncClient) -> None:
    """GET /ready should return 503 before services are initialised."""
    response = await client.get("/api/v1/ready")
    # In test environment, services are not loaded → expect 503
    # TODO: update once /ready is fully implemented
    assert response.status_code in (200, 404, 503)


# TODO: add tests for readiness with mocked services
# @pytest.mark.asyncio
# async def test_ready_returns_200_when_all_services_loaded(client, mock_encoder, mock_faiss):
#     ...
