"""
GeoSense-X — Scene Catalog Pydantic Schemas
=============================================
Request and response models for the scene catalog endpoints (/scenes/*).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from models.schemas.common import GeoPoint, ModalityEnum, Pagination, SeasonEnum


class SceneMetadata(BaseModel):
    """
    Lightweight metadata attached to every retrieval result.

    Returned inline within each SceneMatch to avoid extra API calls.
    """

    lat: float = Field(..., description="Scene centre latitude (WGS84)")
    lon: float = Field(..., description="Scene centre longitude (WGS84)")
    scene_date: date | None = Field(default=None, description="Acquisition date")
    season: SeasonEnum | None = Field(default=None, description="Acquisition season")
    lclu_class: str = Field(..., description="MODIS land cover class name")

    model_config = {"use_enum_values": True}


class SceneDetail(BaseModel):
    """
    Full scene metadata returned by GET /scenes/{scene_id}.

    Includes all available information about a scene in the catalog.
    """

    scene_id: str = Field(..., description="Unique scene identifier")
    lat: float
    lon: float
    scene_date: date | None = None
    season: SeasonEnum | None = None
    lclu_class: str
    lclu_class_id: int
    has_optical: bool = False
    has_sar: bool = False
    has_multispectral: bool = False
    cloud_cover_pct: float | None = None
    thumbnail_url: str | None = None
    source_dataset: str = "sen12ms"

    model_config = {"use_enum_values": True}


class SceneListResponse(BaseModel):
    """Paginated list of scenes returned by GET /scenes/."""

    items: list[SceneDetail]
    total: int = Field(..., description="Total number of matching scenes")
    page: int
    limit: int
    pages: int = Field(..., description="Total number of pages")


class SceneListRequest(BaseModel):
    """Query parameters for GET /scenes/ list endpoint."""

    modality: ModalityEnum | None = None
    lclu_class: str | None = None
    bbox: str | None = Field(
        default=None,
        description="Bounding box as 'min_lon,min_lat,max_lon,max_lat'",
    )
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    season: SeasonEnum | None = None
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)

    model_config = {"use_enum_values": True}
