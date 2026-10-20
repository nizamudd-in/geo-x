"""
GeoSense-X — Storage Backend Abstraction
==========================================
Provides a unified interface over local filesystem and AWS S3 storage.

The active backend is selected at startup from STORAGE_BACKEND env var.
All heatmap PNGs and thumbnails are written through this abstraction,
so switching from local dev storage to S3 in production requires
only an env var change.

Usage:
    from core.storage import get_storage
    storage = get_storage()
    url = await storage.upload("heatmaps/abc123.png", png_bytes, "image/png")
    data = await storage.download("heatmaps/abc123.png")
"""

from __future__ import annotations

import abc
from pathlib import Path

import aiofiles

from config import settings
from core.exceptions import StorageError
from core.logging import get_logger

logger = get_logger(__name__)


class StorageBackend(abc.ABC):
    """Abstract base class for all storage backends."""

    @abc.abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """
        Upload binary data to storage and return a URL/path.

        Args:
            key: Storage key / object path (e.g., "heatmaps/abc123.png").
            data: Raw bytes to store.
            content_type: MIME type of the data.

        Returns:
            URL or local path string for accessing the stored object.

        Raises:
            StorageError: If the upload fails.
        """

    @abc.abstractmethod
    async def download(self, key: str) -> bytes | None:
        """
        Download binary data by key.

        Args:
            key: Storage key to retrieve.

        Returns:
            Raw bytes if found, None if key does not exist.

        Raises:
            StorageError: If the download fails for reasons other than 404.
        """

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if the key exists in storage."""

    @abc.abstractmethod
    def get_url(self, key: str) -> str:
        """Return the public/accessible URL for a stored key."""


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend for development.

    All objects are stored under a configurable base directory.
    get_url() returns a relative path suitable for use as a static file URL.
    """

    def __init__(self, base_dir: Path) -> None:
        """
        Initialise local storage.

        Args:
            base_dir: Root directory for all stored objects.
        """
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    async def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Write data to local filesystem under base_dir/key."""
        dest = self._base_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiofiles.open(dest, "wb") as f:
                await f.write(data)
            logger.debug("LocalStorage upload", key=key, size=len(data))
            return self.get_url(key)
        except OSError as exc:
            raise StorageError("upload", key, str(exc)) from exc

    async def download(self, key: str) -> bytes | None:
        """Read data from local filesystem."""
        src = self._base_dir / key
        if not src.exists():
            return None
        try:
            async with aiofiles.open(src, "rb") as f:
                return await f.read()
        except OSError as exc:
            raise StorageError("download", key, str(exc)) from exc

    async def exists(self, key: str) -> bool:
        """Check if file exists on local filesystem."""
        return (self._base_dir / key).exists()

    def get_url(self, key: str) -> str:
        """Return static-file URL path for the key."""
        # TODO: make base URL configurable from settings
        return f"/static/{key}"


class S3Storage(StorageBackend):
    """
    AWS S3 storage backend for production.

    Uses aioboto3 for async S3 operations.
    Objects are stored under s3://BUCKET/PREFIX/key.
    get_url() returns a presigned URL valid for 1 hour.
    """

    def __init__(self) -> None:
        """Initialise S3 storage client."""
        # TODO: implement S3 storage once aioboto3 is configured
        self._bucket = settings.S3_BUCKET
        self._prefix = settings.S3_PREFIX
        self._region = settings.S3_REGION
        self._session = None  # TODO: aioboto3.Session()

    async def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload object to S3."""
        # TODO: implement S3 upload
        # full_key = f"{self._prefix}/{key}"
        # async with self._session.client("s3", region_name=self._region) as s3:
        #     await s3.put_object(Bucket=self._bucket, Key=full_key, Body=data,
        #                         ContentType=content_type)
        # return self.get_url(key)
        raise NotImplementedError("S3Storage.upload not yet implemented")

    async def download(self, key: str) -> bytes | None:
        """Download object from S3."""
        # TODO: implement S3 download
        raise NotImplementedError("S3Storage.download not yet implemented")

    async def exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        # TODO: implement S3 head_object check
        raise NotImplementedError("S3Storage.exists not yet implemented")

    def get_url(self, key: str) -> str:
        """Return S3 object URL."""
        # TODO: generate presigned URL
        full_key = f"{self._prefix}/{key}"
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{full_key}"


# Module-level singleton
_storage_instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """
    Return the configured storage backend singleton.

    Backend is selected from STORAGE_BACKEND env var:
      "local" → LocalStorage (development)
      "s3"    → S3Storage (production)

    Returns:
        Configured StorageBackend instance.
    """
    global _storage_instance
    if _storage_instance is None:
        if settings.STORAGE_BACKEND == "s3":
            _storage_instance = S3Storage()
        else:
            _storage_instance = LocalStorage(Path("backend/static"))
    return _storage_instance
