"""
GeoSense-X — Explanation Pydantic Schemas
==========================================
Request and response models for the Explainable Retrieval Engine (XRE).

These are nested inside RetrievalResult in the retrieval response.
They are also returned standalone from GET /explain/* endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class NarrativeCategoryEnum(str, Enum):
    """Narrative category for the natural language explanation."""
    STRONG_MATCH = "strong_semantic_match"
    PARTIAL_MATCH = "partial_match_with_divergence"
    MODERATE_MATCH = "moderate_semantic_match"
    UNCERTAIN = "uncertain_match"


class AxisVerdictEnum(str, Enum):
    """Quality verdict for a single semantic axis comparison."""
    STRONG_MATCH = "strong match"
    PARTIAL_MATCH = "partial match"
    SIGNIFICANT_DIVERGENCE = "significant divergence"


# --------------------------------------------------------------------------- #
# Component Schemas
# --------------------------------------------------------------------------- #

class AxisScore(BaseModel):
    """
    Pairwise semantic axis comparison between query and result.

    Each field represents the axis score for the query or result,
    and the derived similarity and verdict.
    """

    query_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Axis score for the query image [0=absent, 1=dominant]",
    )
    result_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Axis score for the result scene",
    )
    similarity: float = Field(
        ..., ge=0.0, le=1.0,
        description="Axis similarity: 1 - |query_score - result_score|",
    )
    verdict: AxisVerdictEnum = Field(
        ...,
        description="Human-readable verdict for this axis",
    )
    divergence_direction: str | None = Field(
        default=None,
        description="If diverging: 'query has more X' or 'result has more X'",
    )

    model_config = {"use_enum_values": True}


class SemanticAxisBundle(BaseModel):
    """Four-axis semantic similarity breakdown for a (query, result) pair."""

    vegetation: AxisScore
    waterbody: AxisScore
    urban: AxisScore
    road_network: AxisScore

    @property
    def convergent_axes(self) -> list[str]:
        """Return axis names where similarity > 0.75."""
        # TODO: implement once AxisScore is fully populated
        axes = {"vegetation": self.vegetation, "waterbody": self.waterbody,
                "urban": self.urban, "road_network": self.road_network}
        return [name for name, axis in axes.items() if axis.similarity > 0.75]

    @property
    def divergent_axes(self) -> list[str]:
        """Return axis names where similarity < 0.40."""
        # TODO: implement once AxisScore is fully populated
        axes = {"vegetation": self.vegetation, "waterbody": self.waterbody,
                "urban": self.urban, "road_network": self.road_network}
        return [name for name, axis in axes.items() if axis.similarity < 0.40]


class RegionImportance(BaseModel):
    """Top-attended region from DINOv2 attention map × land cover mask."""

    class_name: str = Field(..., description="LCLU class name of this region")
    attention_density: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalised attention density in this region",
    )
    coverage_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage of scene area covered by this region",
    )


class HeatmapURLs(BaseModel):
    """URLs to the three attention heatmap images for a (query, result) pair."""

    query: str | None = Field(
        default=None,
        description="URL to query image attention heatmap PNG",
    )
    result: str | None = Field(
        default=None,
        description="URL to result image attention heatmap PNG",
    )
    diff: str | None = Field(
        default=None,
        description="URL to diff overlay PNG (red=query-only, blue=result-only, green=shared)",
    )


class SpatialAttributionBundle(BaseModel):
    """Spatial attention attribution for a (query, result) pair."""

    heatmap_urls: HeatmapURLs = Field(
        default_factory=HeatmapURLs,
        description="URLs to generated heatmap images",
    )
    attention_alignment: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="SSIM score between query and result attention patterns",
    )
    alignment_interpretation: str | None = Field(
        default=None,
        description="Human-readable interpretation of alignment score",
    )


# --------------------------------------------------------------------------- #
# Main Explanation Bundle
# --------------------------------------------------------------------------- #

class ExplanationBundle(BaseModel):
    """
    Complete explanation for a single (query, result) retrieval pair.

    Returned inside each SceneMatch in the retrieval response.
    Also returned standalone from GET /explain/{query_id}/{scene_id}.
    """

    semantic_axes: SemanticAxisBundle | None = Field(
        default=None,
        description="Per-axis similarity breakdown",
    )
    spatial_attribution: SpatialAttributionBundle = Field(
        default_factory=SpatialAttributionBundle,
        description="Attention heatmaps and alignment score",
    )
    region_importance: list[RegionImportance] = Field(
        default_factory=list,
        max_length=3,
        description="Top-3 attended regions in the scene",
    )
    narrative_category: NarrativeCategoryEnum | None = Field(
        default=None,
        description="Explanation narrative category (A-D)",
    )
    natural_language: str | None = Field(
        default=None,
        description="Template-generated natural language explanation",
    )
    explanation_version: str = Field(
        default="1.0",
        description="XRE version that generated this explanation",
    )
    generated_at: datetime | None = Field(
        default=None,
        description="When this explanation was computed",
    )
    from_cache: bool = Field(
        default=False,
        description="True if this explanation was served from cache",
    )

    model_config = {"use_enum_values": True}
