"""
GeoSense-X — Confidence Pydantic Schemas
==========================================
Request and response models for the confidence scoring subsystem.

These schemas are embedded inside RetrievalResult and ExplanationBundle —
they are not exposed as standalone API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.schemas.common import ConfidenceTierEnum


class ConfidenceBundle(BaseModel):
    """
    Complete confidence assessment for a single retrieval result.

    Combines calibrated probability, uncertainty estimate, similarity gap,
    and modality consistency into a single composite score with tier.
    """

    raw_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw cosine similarity from FAISS search",
    )
    calibrated_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P(relevant | similarity, lclu_class) after temperature scaling",
    )
    uncertainty: float = Field(
        ...,
        ge=0.0,
        description="MC Dropout embedding uncertainty scalar (std across T passes)",
    )
    similarity_gap: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Margin between this result's similarity and top-1 similarity",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted composite confidence score",
    )
    confidence_tier: ConfidenceTierEnum = Field(
        ...,
        description="Human-readable confidence tier",
    )
    uncertainty_bound: str = Field(
        ...,
        description="1-sigma uncertainty bound string, e.g. '±0.04'",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Warning flags: cross_modal_gap, sparse_region, etc.",
    )

    model_config = {"use_enum_values": True}


class MCDropoutResult(BaseModel):
    """
    Internal result from Monte Carlo Dropout uncertainty estimation.

    Not returned directly in API responses — used internally by
    ConfidencePipeline to compute the final ConfidenceBundle.
    """

    # TODO: use numpy array type once numpy types are configured for Pydantic
    uncertainty_scalar: float = Field(
        ...,
        description="Mean std deviation across embedding dimensions",
    )
    n_passes: int = Field(
        ...,
        description="Number of stochastic forward passes performed",
    )


class CalibrationParams(BaseModel):
    """
    Temperature scaling parameters for a single LCLU class.

    Loaded from the `calibration_records` table at startup.
    """

    lclu_class: str
    temperature: float = Field(..., gt=0.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    ece_after: float | None = None
