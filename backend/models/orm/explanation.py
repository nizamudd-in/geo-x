"""
GeoSense-X — SceneExplanation ORM Model
=========================================
Maps to the `scene_explanations` table.

Stores precomputed per-scene explanation signals computed once at
ingestion time. This is the Level 2 (PostgreSQL) cache described in
the caching strategy — permanent, invalidated only on model update.

Including the 18×18 compressed attention grid in the DB row enables
fast SSIM alignment computation at query time without S3 round-trips.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.orm.base import Base


class SceneExplanation(Base):
    """
    Precomputed explanation signals for a single satellite scene.

    Populated by the ingestion pipeline (workers/ingestion_tasks.py).
    Invalidated (model_version set to 'STALE') when a new model is deployed.
    """

    __tablename__ = "scene_explanations"

    # ------------------------------------------------------------------ #
    # Primary Key (also FK to scenes)
    # ------------------------------------------------------------------ #
    scene_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("scenes.scene_id", ondelete="CASCADE"),
        primary_key=True,
        doc="Scene this explanation belongs to.",
    )

    # ------------------------------------------------------------------ #
    # Model Version
    # ------------------------------------------------------------------ #
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Model version this explanation was computed with. "
            "Set to 'STALE' when a new model is deployed.",
    )

    # ------------------------------------------------------------------ #
    # Semantic Axis Scores
    # ------------------------------------------------------------------ #
    axis_vegetation: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Vegetation axis score [0, 1] from linear probe.",
    )
    axis_waterbody: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Waterbody axis score [0, 1] from linear probe.",
    )
    axis_urban: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
        doc="Urban density axis score [0, 1] from linear probe.",
    )
    axis_road: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Road network axis score [0, 1] from linear probe.",
    )

    # ------------------------------------------------------------------ #
    # Spectral Index Summaries
    # ------------------------------------------------------------------ #
    ndvi_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndvi_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndwi_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndbi_mean: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ------------------------------------------------------------------ #
    # Coverage Profiles (stored as JSON — no ARRAY for SQLite compat)
    # ------------------------------------------------------------------ #
    vegetation_profile_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON [dense_forest, moderate_veg, sparse_veg, non_veg].",
    )
    water_profile_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON [open_water, wetland, moist_soil, dry_land].",
    )
    urban_profile_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON [dense_urban, sparse_urban, bare_soil, vegetated].",
    )
    road_features_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON [edge_density, line_count_norm, mean_len_norm, orient_entropy].",
    )

    # ------------------------------------------------------------------ #
    # Attention Grid (compressed)
    # ------------------------------------------------------------------ #
    attention_grid_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON serialisation of 18×18 compressed attention grid.",
    )
    attention_heatmap_s3: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="S3 URI or local path to full-res 256×256 attention heatmap PNG.",
    )

    # ------------------------------------------------------------------ #
    # Region Importance
    # ------------------------------------------------------------------ #
    region_importance: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="List of top-3 attended regions: [{class, attention_density, coverage_pct}].",
    )

    # ------------------------------------------------------------------ #
    # Uncertainty
    # ------------------------------------------------------------------ #
    embedding_uncertainty: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="MC Dropout uncertainty scalar computed at ingestion (T=20 passes).",
    )

    # ------------------------------------------------------------------ #
    # Timestamps
    # ------------------------------------------------------------------ #
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    compute_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<SceneExplanation scene={self.scene_id!r} "
            f"version={self.model_version!r}>"
        )
