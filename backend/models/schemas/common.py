"""
GeoSense-X — Shared Pydantic Schema Types
==========================================
Common types reused across multiple schema modules.

Import these instead of redeclaring the same types in every schema file.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class ModalityEnum(str, Enum):
    """Valid satellite image modalities."""
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    ALL = "all"


class SeasonEnum(str, Enum):
    """Seasons used in SEN12MS dataset."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


class ExplanationDepthEnum(str, Enum):
    """Controls how much explanation work is done synchronously."""
    FAST = "fast"   # axis scores + confidence + narrative only (<80ms overhead)
    FULL = "full"   # + attention heatmaps (<500ms overhead)


class ConfidenceTierEnum(str, Enum):
    """Human-readable confidence tier labels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


# --------------------------------------------------------------------------- #
# Geo Types
# --------------------------------------------------------------------------- #

class GeoPoint(BaseModel):
    """WGS84 geographic coordinate."""
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")


class BBoxParam(BaseModel):
    """Bounding box filter in WGS84 coordinates."""
    min_lon: float = Field(..., ge=-180.0, le=180.0)
    min_lat: float = Field(..., ge=-90.0, le=90.0)
    max_lon: float = Field(..., ge=-180.0, le=180.0)
    max_lat: float = Field(..., ge=-90.0, le=90.0)

    def as_list(self) -> list[float]:
        """Return bbox as [min_lon, min_lat, max_lon, max_lat]."""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #

class Pagination(BaseModel):
    """Pagination parameters for list endpoints."""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    limit: int = Field(default=20, ge=1, le=100, description="Results per page")

    @property
    def offset(self) -> int:
        """Return SQL OFFSET value for this page."""
        return (self.page - 1) * self.limit


# --------------------------------------------------------------------------- #
# Filter Parameters
# --------------------------------------------------------------------------- #

class FilterParams(BaseModel):
    """Optional filters applied to retrieval search results."""

    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat]",
    )
    date_from: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date filter (YYYY-MM-DD)",
    )
    date_to: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="End date filter (YYYY-MM-DD)",
    )
    season: SeasonEnum | None = Field(default=None, description="Season filter")
    target_modality: ModalityEnum | None = Field(
        default=None,
        description="Override target modality (for cross-modal endpoint)",
    )


# --------------------------------------------------------------------------- #
# Error Response
# --------------------------------------------------------------------------- #

class ErrorResponse(BaseModel):
    """Standard error response body."""
    error: str = Field(..., description="Error code, e.g. 'SCENE_NOT_FOUND'")
    detail: str = Field(..., description="Human-readable error description")
    request_id: str | None = Field(default=None, description="Request correlation ID")
