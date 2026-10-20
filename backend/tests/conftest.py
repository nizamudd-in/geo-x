"""
GeoSense-X — Pytest Configuration & Shared Fixtures
=====================================================
Provides async test client, test database session, and mock singletons
used across all test modules.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from main import create_app


@pytest.fixture(scope="session")
def app():
    """Create FastAPI test application instance."""
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    """
    Async HTTP test client that calls the FastAPI app directly (no network).
    Resets between each test.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def sample_optical_image_bytes() -> bytes:
    """Return minimal valid JPEG bytes for testing optical preprocessing."""
    # TODO: generate a real minimal JPEG (1×1 pixel)
    # from PIL import Image
    # import io
    # img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    # buf = io.BytesIO()
    # img.save(buf, format="JPEG")
    # return buf.getvalue()
    return b"\xff\xd8\xff\xe0"  # JPEG header stub


@pytest.fixture
def sample_embedding() -> "np.ndarray":
    """Return a random L2-normalised 128-d embedding for testing."""
    import numpy as np
    emb = np.random.randn(128).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


# TODO: add async DB session fixture once test DB is configured
# @pytest_asyncio.fixture
# async def test_db():
#     from core.database import get_engine
#     from models.orm.base import Base
#     engine = create_async_engine("sqlite+aiosqlite:///:memory:")
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#     yield engine
#     await engine.dispose()
