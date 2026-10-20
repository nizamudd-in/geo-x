"""
GeoSense-X — Retrieval API Endpoints
======================================
Three POST endpoints:

  POST /api/v1/retrieve/optical       →  optical → optical
  POST /api/v1/retrieve/sar           →  SAR → SAR
  POST /api/v1/retrieve/cross-modal   →  any modality → any modality

All endpoints accept multipart/form-data with an image file upload.
Heavy work (encode + FAISS search) runs synchronously on the thread pool
because FastAPI dispatches sync functions via executor automatically.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from config import settings
from schemas.retrieval import RetrievalResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/retrieve", tags=["retrieval"])

_VALID_SOURCE = {"optical", "sar"}
_VALID_TARGET = {"optical", "sar", "all"}
_MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


async def _read_upload(file: UploadFile) -> bytes:
    """Read and validate an uploaded image file."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > _MAX_BYTES:
        size_mb = len(data) / 1_048_576
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {settings.MAX_FILE_SIZE_MB} MB.",
        )
    return data


def _get_service(request: Request):
    """Return the RetrievalService singleton from app state."""
    svc = getattr(request.app.state, "retrieval", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialised.")
    if not request.app.state.faiss.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "FAISS indexes are not loaded. "
                "Run build_indexes() after generating embeddings, "
                "then restart the server."
            ),
        )
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/optical",
    response_model=RetrievalResponse,
    summary="Optical → Optical retrieval",
    description=(
        "Upload an RGB optical satellite image and retrieve the top-K most "
        "visually similar optical scenes from the indexed catalog."
    ),
)
async def retrieve_optical(
    request: Request,
    image: UploadFile = File(..., description="RGB satellite image (JPEG / PNG / GeoTIFF)"),
    top_k: int = Form(
        default=10,
        ge=1,
        le=10,
        description="Number of results to return (1–10)",
    ),
) -> RetrievalResponse:
    data = await _read_upload(image)
    svc  = _get_service(request)
    return await run_in_threadpool(svc.retrieve, data, "optical", "optical", top_k)


@router.post(
    "/sar",
    response_model=RetrievalResponse,
    summary="SAR → SAR retrieval",
    description=(
        "Upload a SAR image (GeoTIFF with VV/VH bands, or grayscale fallback) "
        "and retrieve the top-K most similar SAR scenes."
    ),
)
async def retrieve_sar(
    request: Request,
    image: UploadFile = File(..., description="SAR image (2-band GeoTIFF preferred)"),
    top_k: int = Form(default=10, ge=1, le=10),
) -> RetrievalResponse:
    data = await _read_upload(image)
    svc  = _get_service(request)
    return await run_in_threadpool(svc.retrieve, data, "sar", "sar", top_k)


@router.post(
    "/cross-modal",
    response_model=RetrievalResponse,
    summary="Cross-modal retrieval (optical ↔ SAR)",
    description=(
        "Upload an image in any supported modality and retrieve results in a "
        "different modality. The core GeoSense-X feature. "
        "Example: upload an optical image, get back matching SAR scenes."
    ),
)
async def retrieve_cross_modal(
    request: Request,
    image: UploadFile = File(..., description="Query image (any supported modality)"),
    source_modality: str = Form(
        ...,
        description="Modality of the uploaded image: optical | sar",
    ),
    target_modality: str = Form(
        ...,
        description="Target retrieval modality: optical | sar | all",
    ),
    top_k: int = Form(default=10, ge=1, le=10),
) -> RetrievalResponse:
    source_modality = source_modality.strip().lower()
    target_modality = target_modality.strip().lower()

    if source_modality not in _VALID_SOURCE:
        raise HTTPException(
            status_code=422,
            detail=f"source_modality must be one of {_VALID_SOURCE}, got {source_modality!r}.",
        )
    if target_modality not in _VALID_TARGET:
        raise HTTPException(
            status_code=422,
            detail=f"target_modality must be one of {_VALID_TARGET}, got {target_modality!r}.",
        )

    data = await _read_upload(image)
    svc  = _get_service(request)
    return await run_in_threadpool(
        svc.retrieve, data, source_modality, target_modality, top_k
    )
