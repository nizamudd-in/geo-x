"""
GeoSense-X — Retrieval Request / Response Schemas
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SceneMatch(BaseModel):
    """A single retrieved scene with similarity score and metadata."""

    rank: int = Field(..., ge=1, description="Result rank (1 = most similar)")
    scene_id: str
    modality: str = Field(..., description="optical | sar | multispectral")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity [0, 1]")
    thumbnail_url: str | None = None
    lat: float | None = None
    lon: float | None = None
    lclu_class: str | None = None
    season: str | None = None
    scene_date: str | None = None


class RetrievalResponse(BaseModel):
    """Top-K retrieval results for a single query."""

    query_id: str
    source_modality: str
    target_modality: str
    latency_ms: int
    result_count: int
    results: list[SceneMatch]
