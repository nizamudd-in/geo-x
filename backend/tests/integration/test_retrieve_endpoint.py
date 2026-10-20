"""
Integration tests for POST /api/v1/retrieve/* endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_retrieve_optical_requires_image(client: AsyncClient) -> None:
    """POST /retrieve/optical without image should return 422."""
    response = await client.post("/api/v1/retrieve/optical", data={"top_k": 10})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_retrieve_cross_modal_requires_modalities(client: AsyncClient) -> None:
    """POST /retrieve/cross-modal without required modality fields should return 422."""
    response = await client.post(
        "/api/v1/retrieve/cross-modal",
        data={"top_k": 10},
    )
    assert response.status_code == 422


# TODO: add full integration test once pipeline is implemented
# @pytest.mark.asyncio
# async def test_retrieve_optical_returns_top10(
#     client: AsyncClient, sample_optical_image_bytes: bytes
# ) -> None:
#     """POST /retrieve/optical should return 10 results with correct schema."""
#     response = await client.post(
#         "/api/v1/retrieve/optical",
#         files={"image": ("test.jpg", sample_optical_image_bytes, "image/jpeg")},
#         data={"top_k": 10, "explanation_depth": "fast"},
#     )
#     assert response.status_code == 200
#     data = response.json()
#     assert len(data["results"]) == 10
#     assert data["results"][0]["rank"] == 1
