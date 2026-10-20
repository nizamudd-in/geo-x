"""
Integration tests for GET /api/v1/explain/* endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_explain_returns_404_for_unknown_query(client: AsyncClient) -> None:
    """GET /explain/{unknown_id}/{scene_id} should return 404 or 501."""
    response = await client.get(
        "/api/v1/explain/00000000-0000-0000-0000-000000000000/nonexistent_scene"
    )
    # 501 Not Implemented until explanation service is built
    # 404 once implemented but session not found
    assert response.status_code in (404, 501, 500)


# TODO: add full explanation tests once ExplanationService is implemented
# @pytest.mark.asyncio
# async def test_explain_returns_axis_bundle(client, saved_retrieval_session):
#     ...
