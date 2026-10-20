"""
GeoSense-X — Retrieval Pydantic Schemas
=========================================
Request and response models for POST /retrieve/* endpoints.

RetrievalResponse is the primary output of the system — every other schema
module is either nested inside it or supports it.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from models.schemas.common import ExplanationDepthEnum, FilterParams, ModalityEnum
from models.schemas.confidence import ConfidenceBundle
from models.schemas.explanation import ExplanationBundle
from models.schemas.scene import SceneMetadata


# --------------------------------------------------------------------------- #
# Request Schemas
# --------------------------------------------------------------------------- #

class RetrievalRequest(BaseModel):
    """
    Parameters for a retrieval request.

    The image itself is uploaded as a multipart/form-data file field
    (handled by FastAPI UploadFile) — this schema captures the remaining
    form fields.
    """

    source_modality: ModalityEnum = Field(
        ...,
        description="Modality of the uploaded query image",
    )
    target_modality: ModalityEnum = Field(
        default=ModalityEnum.OPTICAL,
        description="Target retrieval modality. Use 'all' for mixed-modality results",
    )
    top_k: int = Field(
        default=10,
        description="Number of results to return",
        ge=1,
        le=10,
    )
    explanation_depth: ExplanationDepthEnum = Field(
        default=ExplanationDepthEnum.FAST,
        description="'fast' returns axes + confidence inline. 'full' adds heatmaps.",
    )
    filters: FilterParams | None = Field(
        default=None,
        description="Optional search filters (bbox, date range, season)",
    )

    model_config = {"use_enum_values": True}


class RefinementAxis(BaseModel):
    """Single axis refinement directive for POST /search/refine."""

    axis: str = Field(
        ...,
        description="Axis to adjust: vegetation | waterbody | urban | road_network",
    )
    direction: str = Field(
        ...,
        description="'more' | 'less' | 'maintain'",
        pattern="^(more|less|maintain)$",
    )
    weight: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Refinement weight multiplier [0.5, 2.0]",
    )


class RefineRequest(BaseModel):
    """Request body for POST /search/refine."""

    query_id: uuid.UUID
    refinements: list[RefinementAxis] = Field(..., min_length=1, max_length=4)
    top_k: int = Field(default=10, ge=1, le=10)


# --------------------------------------------------------------------------- #
# Response Schemas
# --------------------------------------------------------------------------- #

class SceneMatch(BaseModel):
    """
    Single retrieved result scene with explanation and confidence.

    This is the core item in the retrieval response — every top-K result
    is represented as one SceneMatch.
    """

    rank: int = Field(..., ge=1, description="Result rank (1 = most similar)")
    scene_id: str = Field(..., description="Unique scene identifier")
    modality: ModalityEnum = Field(..., description="Modality of this result scene")
    similarity: float = Field(
        ..., ge=0.0, le=1.0,
        description="Cosine similarity to query embedding",
    )
    thumbnail_url: str | None = Field(
        default=None,
        description="URL to the 128×128 JPEG thumbnail",
    )
    metadata: SceneMetadata = Field(..., description="Scene geographic and temporal metadata")
    explanation: ExplanationBundle | None = Field(
        default=None,
        description="Explanation bundle (null if explanation_depth='none')",
    )
    confidence: ConfidenceBundle | None = Field(
        default=None,
        description="Confidence scoring bundle",
    )

    model_config = {"use_enum_values": True}


class RetrievalResponse(BaseModel):
    """
    Complete response from POST /retrieve/* endpoints.

    Contains the query session ID, configuration echo, latency breakdown,
    and ordered list of top-K SceneMatch results.
    """

    query_id: uuid.UUID = Field(..., description="Unique ID for this retrieval session")
    source_modality: ModalityEnum
    target_modality: ModalityEnum
    top_k: int
    latency_ms: int = Field(..., description="Total end-to-end latency in milliseconds")
    encode_latency_ms: int | None = None
    search_latency_ms: int | None = None
    explain_latency_ms: int | None = None
    results: list[SceneMatch] = Field(..., description="Ordered list of top-K results")

    model_config = {"use_enum_values": True}


class BatchExplainRequest(BaseModel):
    """Request body for POST /explain/batch."""

    pairs: list[dict] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of {query_id, scene_id} pairs to explain",
    )
    depth: ExplanationDepthEnum = Field(default=ExplanationDepthEnum.FAST)

    model_config = {"use_enum_values": True}
