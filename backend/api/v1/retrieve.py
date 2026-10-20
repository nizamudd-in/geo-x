"""
GeoSense-X — Retrieval API Endpoints
======================================
Four POST endpoints for image retrieval:

  POST /retrieve/optical       → same-modal optical search
  POST /retrieve/sar           → same-modal SAR search
  POST /retrieve/multispectral → same-modal multispectral search
  POST /retrieve/cross-modal   → cross-modal search (any source → any target)

These endpoints are the primary API surface of the MVP.
All business logic is delegated to RetrievalPipeline — route handlers
only handle HTTP concerns (file upload, schema validation, response serialisation).
"""

from __future__ import annotations

import json
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from api.dependencies import RateLimited, get_encoder, get_faiss
from core.logging import get_logger
from models.schemas.common import ExplanationDepthEnum, ModalityEnum
from models.schemas.retrieval import RetrievalResponse

logger = get_logger(__name__)

router = APIRouter()

# Maximum allowed upload size: 50MB
_MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024


async def _read_image(file: UploadFile) -> bytes:
    """
    Read and validate uploaded image bytes.

    Args:
        file: FastAPI UploadFile from multipart form.

    Returns:
        Raw image bytes.

    Raises:
        PreprocessingError: If file is too large or has invalid content type.
    """
    from core.exceptions import PreprocessingError

    # TODO: validate content-type header more strictly
    data = await file.read()
    if len(data) > _MAX_IMAGE_SIZE_BYTES:
        size_mb = len(data) / 1024 / 1024
        raise PreprocessingError(
            f"Image too large ({size_mb:.1f}MB). Maximum is 50MB."
        )
    return data


@router.post(
    "/optical",
    response_model=RetrievalResponse,
    summary="Optical same-modal retrieval",
    description="Upload an optical (RGB) satellite image and retrieve the "
                "top-K most similar optical scenes from the catalog.",
)
async def retrieve_optical(
    request: Request,
    _rate_limit: RateLimited,
    image: UploadFile = File(..., description="Optical satellite image (JPEG/PNG/GeoTIFF)"),
    top_k: int = Form(default=10, ge=1, le=10),
    explanation_depth: ExplanationDepthEnum = Form(default=ExplanationDepthEnum.FAST),
    filters: str | None = Form(default=None, description="JSON-encoded FilterParams"),
) -> RetrievalResponse:
    """
    Same-modal optical retrieval endpoint.

    Encodes the uploaded optical image, searches the optical FAISS index,
    and returns the top-K most similar optical scenes with explanations.
    """
    image_bytes = await _read_image(image)

    # TODO: call RetrievalPipeline once domain/retrieval/pipeline.py is implemented
    # from domain.retrieval.pipeline import RetrievalPipeline
    # pipeline = RetrievalPipeline(
    #     encoder=await get_encoder(request),
    #     faiss=await get_faiss(request),
    # )
    # return await pipeline.run(
    #     image_bytes=image_bytes,
    #     source_modality="optical",
    #     target_modality="optical",
    #     top_k=top_k,
    #     explanation_depth=explanation_depth,
    #     filters=json.loads(filters) if filters else None,
    # )

    # Placeholder response — remove once pipeline is implemented
    raise NotImplementedError("Optical retrieval pipeline not yet implemented")


@router.post(
    "/sar",
    response_model=RetrievalResponse,
    summary="SAR same-modal retrieval",
    description="Upload a SAR image and retrieve the top-K most similar "
                "SAR scenes from the catalog.",
)
async def retrieve_sar(
    request: Request,
    _rate_limit: RateLimited,
    image: UploadFile = File(..., description="SAR image (GeoTIFF with VV/VH bands)"),
    top_k: int = Form(default=10, ge=1, le=10),
    explanation_depth: ExplanationDepthEnum = Form(default=ExplanationDepthEnum.FAST),
    filters: str | None = Form(default=None),
) -> RetrievalResponse:
    """Same-modal SAR retrieval endpoint."""
    image_bytes = await _read_image(image)

    # TODO: implement SAR retrieval pipeline call
    raise NotImplementedError("SAR retrieval pipeline not yet implemented")


@router.post(
    "/multispectral",
    response_model=RetrievalResponse,
    summary="Multispectral same-modal retrieval",
    description="Upload a 13-band multispectral image and retrieve the "
                "top-K most similar multispectral scenes.",
)
async def retrieve_multispectral(
    request: Request,
    _rate_limit: RateLimited,
    image: UploadFile = File(..., description="Multispectral GeoTIFF (13 bands)"),
    top_k: int = Form(default=10, ge=1, le=10),
    explanation_depth: ExplanationDepthEnum = Form(default=ExplanationDepthEnum.FAST),
    filters: str | None = Form(default=None),
) -> RetrievalResponse:
    """Same-modal multispectral retrieval endpoint."""
    image_bytes = await _read_image(image)

    # TODO: implement multispectral retrieval pipeline call
    raise NotImplementedError("Multispectral retrieval pipeline not yet implemented")


@router.post(
    "/cross-modal",
    response_model=RetrievalResponse,
    summary="Cross-modal retrieval (core MVP feature)",
    description="Upload an image in any modality and retrieve results in a "
                "different modality. E.g., optical query → SAR results. "
                "This is the primary demonstration endpoint.",
)
async def retrieve_cross_modal(
    request: Request,
    _rate_limit: RateLimited,
    image: UploadFile = File(..., description="Query image (any modality)"),
    source_modality: ModalityEnum = Form(
        ...,
        description="Modality of the uploaded image",
    ),
    target_modality: ModalityEnum = Form(
        ...,
        description="Target retrieval modality (must differ from source for cross-modal)",
    ),
    top_k: int = Form(default=10, ge=1, le=10),
    explanation_depth: ExplanationDepthEnum = Form(default=ExplanationDepthEnum.FAST),
    filters: str | None = Form(default=None),
) -> RetrievalResponse:
    """
    Cross-modal retrieval endpoint — the core MVP feature.

    Encodes the query image in its source modality, then searches the
    shared cross-modal FAISS index to retrieve scenes in the target modality.

    All four cross-modal directions are supported:
      optical → SAR
      optical → multispectral
      SAR     → optical
      SAR     → multispectral
      multispectral → optical
      multispectral → SAR
      any → all  (mixed modality results)
    """
    image_bytes = await _read_image(image)
    start_time = time.perf_counter()

    # TODO: call RetrievalPipeline once implemented
    # from domain.retrieval.pipeline import RetrievalPipeline
    # pipeline = RetrievalPipeline(
    #     encoder=await get_encoder(request),
    #     faiss=await get_faiss(request),
    # )
    # result = await pipeline.run(
    #     image_bytes=image_bytes,
    #     source_modality=source_modality,
    #     target_modality=target_modality,
    #     top_k=top_k,
    #     explanation_depth=explanation_depth,
    #     filters=json.loads(filters) if filters else None,
    # )
    # logger.info(
    #     "cross_modal_retrieval",
    #     query_id=str(result.query_id),
    #     source=source_modality,
    #     target=target_modality,
    #     latency_ms=result.latency_ms,
    # )
    # return result

    raise NotImplementedError("Cross-modal retrieval pipeline not yet implemented")
