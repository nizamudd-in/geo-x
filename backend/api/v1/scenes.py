"""
GeoSense-X — Scene Catalog Endpoints
======================================
Endpoints for browsing and inspecting the satellite scene catalog.

  GET /scenes/                     → paginated scene list with filters
  GET /scenes/{scene_id}           → full scene metadata
  GET /scenes/{scene_id}/thumbnail → 128×128 JPEG thumbnail
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import StreamingResponse

from core.logging import get_logger
from models.schemas.scene import SceneDetail, SceneListResponse

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=SceneListResponse,
    summary="List scenes with optional filters",
    description="Returns a paginated list of satellite scenes. "
                "Filter by modality, LCLU class, bounding box, or date range.",
)
async def list_scenes(
    request: Request,
    modality: str | None = Query(default=None, description="Filter by modality"),
    lclu_class: str | None = Query(default=None, description="Filter by land cover class"),
    bbox: str | None = Query(
        default=None,
        description="Bounding box: 'min_lon,min_lat,max_lon,max_lat'",
    ),
    date_from: str | None = Query(default=None, description="Start date (YYYY-MM-DD)"),
    date_to: str | None = Query(default=None, description="End date (YYYY-MM-DD)"),
    season: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> SceneListResponse:
    """
    List scenes from the catalog with optional filtering and pagination.

    Results are ordered by scene_date descending.
    Bounding box filter uses PostGIS ST_Within for spatial accuracy.
    """
    # TODO: implement via scene_repo.list_scenes()
    # from repositories.scene_repo import SceneRepository
    # repo = SceneRepository(db)
    # scenes, total = await repo.list_scenes(filters, pagination)
    # return SceneListResponse(items=scenes, total=total, page=page, limit=limit, ...)
    raise NotImplementedError("Scene listing not yet implemented")


@router.get(
    "/{scene_id}",
    response_model=SceneDetail,
    summary="Get scene metadata",
    description="Returns full metadata for a single scene including "
                "geographic coordinates, acquisition date, and available modalities.",
)
async def get_scene(
    request: Request,
    scene_id: str = Path(..., description="Scene identifier"),
) -> SceneDetail:
    """
    Retrieve full metadata for a single scene.

    Looks up the scene by scene_id in the scenes table.
    """
    # TODO: implement via scene_repo.get_by_id()
    # from repositories.scene_repo import SceneRepository
    # scene = await SceneRepository(db).get_by_id(scene_id)
    # if scene is None:
    #     raise SceneNotFoundError(scene_id)
    # return SceneDetail.model_validate(scene)
    raise NotImplementedError("Scene detail not yet implemented")


@router.get(
    "/{scene_id}/thumbnail",
    summary="Get scene thumbnail image",
    description="Returns a 128×128 JPEG thumbnail for the given scene. "
                "Generated at ingestion time and cached on local filesystem or S3.",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "JPEG thumbnail"},
        404: {"description": "Scene not found or thumbnail not available"},
    },
)
async def get_thumbnail(
    request: Request,
    scene_id: str = Path(..., description="Scene identifier"),
) -> StreamingResponse:
    """
    Serve a pre-computed 128×128 JPEG thumbnail.

    Lookup order:
      1. Local filesystem: static/thumbnails/{scene_id}.jpg
      2. S3 bucket (if configured)
      3. On-demand generation (slow path, cached afterward)
    """
    # TODO: implement via thumbnail_service
    # from services.thumbnail_service import ThumbnailService
    # svc = ThumbnailService()
    # png_bytes = await svc.get_thumbnail(scene_id)
    # if png_bytes is None:
    #     raise SceneNotFoundError(scene_id)
    # return StreamingResponse(iter([png_bytes]), media_type="image/jpeg")
    raise NotImplementedError("Thumbnail service not yet implemented")
