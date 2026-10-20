"""
GeoSense-X — Explanation API Endpoints
========================================
Endpoints to fetch structured explanations for retrieval results.

  GET /explain/{query_id}/{scene_id}            → full ExplanationBundle
  GET /explain/{query_id}/{scene_id}/axes       → semantic axes only
  GET /explain/{query_id}/{scene_id}/confidence → confidence bundle only
  GET /explain/{query_id}/{scene_id}/heatmap/query  → PNG StreamingResponse
  GET /explain/{query_id}/{scene_id}/heatmap/result → PNG StreamingResponse
  GET /explain/{query_id}/{scene_id}/heatmap/diff   → PNG StreamingResponse
  POST /explain/batch                           → batch explanations
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Path, Request
from fastapi.responses import Response, StreamingResponse

from core.logging import get_logger
from models.schemas.confidence import ConfidenceBundle
from models.schemas.explanation import ExplanationBundle, SemanticAxisBundle
from models.schemas.retrieval import BatchExplainRequest

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/{query_id}/{scene_id}",
    response_model=ExplanationBundle,
    summary="Get full explanation for a retrieval result",
    description="Returns the complete ExplanationBundle for a (query, result) pair. "
                "Returns 202 Accepted if the explanation is still being computed asynchronously.",
)
async def get_explanation(
    request: Request,
    query_id: uuid.UUID = Path(...),
    scene_id: str = Path(...),
) -> ExplanationBundle:
    """
    Retrieve the full explanation bundle for a (query_id, scene_id) pair.

    Checks Redis cache first, then DB, then triggers async computation if missing.
    """
    # TODO: implement cache lookup via CacheService
    # cache = CacheService(request.app.state.redis)
    # bundle = await cache.get_json(f"expl:{query_id}:{scene_id}")
    # if bundle:
    #     return ExplanationBundle(**bundle)

    # TODO: check DB retrieval_results table for stored explanation
    # result = await result_repo.get_result(query_id, scene_id, db)
    # if result and result.natural_language:
    #     return build_bundle_from_orm(result)

    # TODO: trigger async Celery task and return 202
    # from workers.heatmap_tasks import generate_heatmap_pair
    # task = generate_heatmap_pair.delay(str(query_id), scene_id)
    # raise ExplanationUnavailable(task_id=task.id)

    raise NotImplementedError("Explanation service not yet implemented")


@router.get(
    "/{query_id}/{scene_id}/axes",
    response_model=SemanticAxisBundle,
    summary="Get semantic axes comparison only",
    description="Fast endpoint — returns only semantic axis scores from cache. "
                "No heatmap computation required.",
)
async def get_axes(
    request: Request,
    query_id: uuid.UUID = Path(...),
    scene_id: str = Path(...),
) -> SemanticAxisBundle:
    """Return just the semantic axis comparison for a result pair."""
    # TODO: implement fast cache lookup for axis data only
    raise NotImplementedError("Axis scoring not yet implemented")


@router.get(
    "/{query_id}/{scene_id}/confidence",
    response_model=ConfidenceBundle,
    summary="Get confidence bundle only",
    description="Returns only the confidence scoring for a result.",
)
async def get_confidence(
    request: Request,
    query_id: uuid.UUID = Path(...),
    scene_id: str = Path(...),
) -> ConfidenceBundle:
    """Return just the confidence bundle for a result pair."""
    # TODO: implement cache lookup for confidence data
    raise NotImplementedError("Confidence service not yet implemented")


@router.get(
    "/{query_id}/{scene_id}/heatmap/{view}",
    summary="Get attention heatmap image",
    description="Returns a PNG heatmap image. "
                "view: 'query' | 'result' | 'diff'. "
                "Returns 404 if not yet computed (poll or use WebSocket).",
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG heatmap image"},
        404: {"description": "Heatmap not yet available"},
        202: {"description": "Heatmap being computed"},
    },
)
async def get_heatmap(
    request: Request,
    query_id: uuid.UUID = Path(...),
    scene_id: str = Path(...),
    view: str = Path(..., pattern="^(query|result|diff)$"),
) -> StreamingResponse:
    """
    Return a PNG heatmap as a streaming response.

    Looks up the heatmap in Redis cache (key: heatmap:{qid}:{sid}:{view}).
    Returns 404 if not computed — client should poll or use WebSocket events.
    """
    # TODO: implement cache lookup for heatmap PNG bytes
    # cache_key = f"heatmap:{query_id}:{scene_id}:{view[0]}"
    # png_bytes = await cache.get(cache_key)
    # if png_bytes is None:
    #     raise HTTPException(status_code=404, detail="Heatmap not yet computed")
    # return StreamingResponse(iter([png_bytes]), media_type="image/png")

    raise NotImplementedError("Heatmap service not yet implemented")


@router.post(
    "/batch",
    response_model=list[ExplanationBundle],
    summary="Batch explanation request",
    description="Request explanations for up to 20 (query_id, scene_id) pairs. "
                "Returns fast-mode explanations for all pairs.",
)
async def batch_explain(
    request: Request,
    body: BatchExplainRequest,
) -> list[ExplanationBundle]:
    """
    Request explanations for multiple result pairs in one call.

    Useful for pre-fetching explanations for all top-K results
    before the user expands any single result.
    """
    # TODO: implement batch explanation via ExplanationService
    raise NotImplementedError("Batch explanation not yet implemented")
