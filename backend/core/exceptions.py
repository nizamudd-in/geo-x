"""
GeoSense-X — Custom Exception Hierarchy
========================================
All application-level exceptions live here.

Design rules:
  - Every custom exception subclasses GeoSenseException (which extends HTTPException).
  - Each exception maps to a specific HTTP status code set at class definition.
  - Route handlers re-raise these exceptions; main.py exception handlers convert
    them to structured JSON responses.
  - Never raise plain HTTPException from domain or service layers.

Usage:
    from core.exceptions import SceneNotFoundError
    raise SceneNotFoundError(scene_id="abc123")
"""

from __future__ import annotations

from fastapi import HTTPException


class GeoSenseException(HTTPException):
    """
    Base class for all GeoSense-X application exceptions.

    Extends FastAPI's HTTPException so it is automatically handled
    by both FastAPI's default handler and our custom handlers.
    Subclasses set a default status_code at class level.
    """

    status_code: int = 500

    def __init__(self, detail: str = "An unexpected error occurred") -> None:
        super().__init__(status_code=self.status_code, detail=detail)


# --------------------------------------------------------------------------- #
# 4xx Client Errors
# --------------------------------------------------------------------------- #


class PreprocessingError(GeoSenseException):
    """
    Raised when an uploaded image cannot be preprocessed.

    Causes: unsupported format, corrupted file, wrong number of bands,
    image dimensions too small, GeoTIFF CRS mismatch.
    HTTP 400 Bad Request.
    """

    status_code = 400

    def __init__(self, detail: str = "Image preprocessing failed") -> None:
        # TODO: add modality hint to detail message
        super().__init__(detail=detail)


class InvalidModalityError(GeoSenseException):
    """
    Raised when an unrecognised modality string is provided.

    HTTP 400 Bad Request.
    """

    status_code = 400

    def __init__(self, modality: str) -> None:
        super().__init__(
            detail=f"Unknown modality '{modality}'. "
            "Valid values: optical, sar, multispectral, all"
        )


class SceneNotFoundError(GeoSenseException):
    """
    Raised when a scene_id does not exist in the catalog.

    HTTP 404 Not Found.
    """

    status_code = 404

    def __init__(self, scene_id: str) -> None:
        super().__init__(detail=f"Scene '{scene_id}' not found in catalog")


class EmbeddingNotFound(GeoSenseException):
    """
    Raised when no embedding has been computed for a (scene, modality) pair.

    HTTP 404 Not Found.
    """

    status_code = 404

    def __init__(self, scene_id: str, modality: str) -> None:
        super().__init__(
            detail=f"Embedding not found for scene '{scene_id}' modality '{modality}'. "
            "Run the embedding computation pipeline first."
        )


class RateLimitExceeded(GeoSenseException):
    """
    Raised when a client exceeds their per-minute request quota.

    HTTP 429 Too Many Requests. The exception handler adds Retry-After header.
    """

    status_code = 429

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        super().__init__(
            detail=f"Rate limit of {limit} requests per {window_seconds}s exceeded"
        )


class AuthenticationError(GeoSenseException):
    """
    Raised when API key is missing or invalid.

    HTTP 401 Unauthorized.
    """

    status_code = 401

    def __init__(self, detail: str = "Invalid or missing API key") -> None:
        super().__init__(detail=detail)


class ForbiddenError(GeoSenseException):
    """
    Raised when an authenticated user lacks permission for an action.

    HTTP 403 Forbidden.
    """

    status_code = 403

    def __init__(self, detail: str = "Insufficient permissions") -> None:
        super().__init__(detail=detail)


# --------------------------------------------------------------------------- #
# 5xx Server / Infrastructure Errors
# --------------------------------------------------------------------------- #


class ModelNotLoadedError(GeoSenseException):
    """
    Raised when a request arrives before the ML model has finished loading.

    HTTP 503 Service Unavailable. The readiness probe returns 503 until resolved.
    """

    status_code = 503

    def __init__(self) -> None:
        super().__init__(
            detail="ML model is not yet loaded. Retry after /ready returns 200."
        )


class IndexNotLoadedError(GeoSenseException):
    """
    Raised when a FAISS index has not been loaded or is being rebuilt.

    HTTP 503 Service Unavailable.
    """

    status_code = 503

    def __init__(self, modality: str = "unknown") -> None:
        super().__init__(
            detail=f"FAISS index for modality '{modality}' is not available. "
            "Check index build status at /admin/index/status."
        )


class StorageError(GeoSenseException):
    """
    Raised when S3 or local filesystem operations fail.

    HTTP 500 Internal Server Error.
    """

    status_code = 500

    def __init__(self, operation: str, key: str, cause: str = "") -> None:
        super().__init__(
            detail=f"Storage {operation} failed for key '{key}'. {cause}".strip()
        )


class ExplanationUnavailable(GeoSenseException):
    """
    Raised when an explanation has been queued but not yet computed.

    HTTP 202 Accepted. Response body includes task_id for polling.
    Callers should poll GET /explain/{query_id}/{scene_id} until 200.
    """

    status_code = 202

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(
            detail=f"Explanation is being computed asynchronously. Task: {task_id}"
        )


class DatabaseError(GeoSenseException):
    """
    Raised when a database operation fails unexpectedly.

    HTTP 500 Internal Server Error. Do not expose raw DB errors to clients.
    """

    status_code = 500

    def __init__(self, detail: str = "A database error occurred") -> None:
        # TODO: add correlation ID for log tracing
        super().__init__(detail=detail)
