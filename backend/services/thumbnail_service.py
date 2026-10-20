"""
GeoSense-X — Thumbnail Service
=================================
Serves precomputed 128×128 JPEG thumbnails for scene IDs.

Lookup order:
  1. Local filesystem: static/thumbnails/{scene_id}.jpg
  2. S3 bucket (if STORAGE_BACKEND=s3)
  3. On-demand generation from raw scene data (slow path, cached afterward)
"""

from __future__ import annotations

from pathlib import Path

from config import settings
from core.logging import get_logger
from core.storage import get_storage

logger = get_logger(__name__)


class ThumbnailService:
    """Handles thumbnail retrieval and on-demand generation."""

    def __init__(self) -> None:
        """Initialise the thumbnail service."""
        self._thumbnails_dir = settings.THUMBNAILS_DIR
        self._storage = get_storage()

    async def get_thumbnail(self, scene_id: str) -> bytes | None:
        """
        Retrieve thumbnail bytes for a scene.

        Tries local filesystem first, then S3, then generates on-demand.

        Args:
            scene_id: Scene identifier.

        Returns:
            JPEG bytes or None if scene does not exist.
        """
        # Step 1: Check local filesystem
        local_path = self._thumbnails_dir / f"{scene_id}.jpg"
        if local_path.exists():
            return local_path.read_bytes()

        # Step 2: Check storage backend (S3 if configured)
        storage_key = f"thumbnails/{scene_id}.jpg"
        data = await self._storage.download(storage_key)
        if data is not None:
            # Cache locally for next request
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            return data

        # Step 3: Generate on-demand
        logger.debug("Thumbnail not found, generating on-demand", scene_id=scene_id)
        generated = await self._generate_thumbnail(scene_id)
        if generated is not None:
            # Cache for future requests
            local_path.write_bytes(generated)
            await self._storage.upload(storage_key, generated, "image/jpeg")
        return generated

    async def _generate_thumbnail(self, scene_id: str) -> bytes | None:
        """
        Generate a 128×128 JPEG thumbnail for a scene on-demand.

        Reads the optical bands from HDF5, normalises to uint8,
        resizes to 128×128, and encodes as JPEG.

        Args:
            scene_id: Scene identifier.

        Returns:
            JPEG bytes or None if scene data not found.
        """
        # TODO: implement on-demand generation
        # from PIL import Image
        # import h5py, io
        # Locate scene in HDF5
        # Load optical bands (or SAR pseudo-RGB if no optical)
        # Normalise and resize
        # Encode as JPEG
        logger.warning("On-demand thumbnail generation not yet implemented", scene_id=scene_id)
        return None
