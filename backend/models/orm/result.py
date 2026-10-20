"""
GeoSense-X — RetrievalResult ORM Model
=========================================
Maps to the `retrieval_results` table.

One row per (session, result scene) pair. Stores the full explanation
and confidence bundle so the session can be replayed from the DB
without re-running the ML pipeline.

User feedback (relevant / not_relevant / partially_relevant) is also
stored here and used for future model calibration.
"""

from __future__ import annotations

import uuid

from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.orm.base import Base, TimestampMixin


class RetrievalResult(TimestampMixin, Base):
    """
    Single result entry within a retrieval session.

    Stores similarity score, full explanation breakdown, confidence
    bundle, and optional user feedback for every result returned
    to the client.
    """

    __tablename__ = "retrieval_results"

    __table_args__ = (
        UniqueConstraint("query_id", "rank", name="uq_results_query_rank"),
        UniqueConstraint("query_id", "scene_id", name="uq_results_query_scene"),
    )

    # ------------------------------------------------------------------ #
    # Primary Key
    # ------------------------------------------------------------------ #
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------ #
    # Foreign Keys
    # ------------------------------------------------------------------ #
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("retrieval_sessions.query_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scene_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("scenes.scene_id"),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #
    rank: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        doc="Result rank within the session (1 = most similar).",
    )
    modality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Modality of this result scene.",
    )
    cosine_similarity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="L2-normalised cosine similarity score [0, 1].",
    )

    # ------------------------------------------------------------------ #
    # Semantic Axis Similarities
    # ------------------------------------------------------------------ #
    axis_sim_vegetation: Mapped[float | None] = mapped_column(Float, nullable=True)
    axis_sim_waterbody: Mapped[float | None] = mapped_column(Float, nullable=True)
    axis_sim_urban: Mapped[float | None] = mapped_column(Float, nullable=True)
    axis_sim_road: Mapped[float | None] = mapped_column(Float, nullable=True)
    axis_verdicts: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="Per-axis verdict strings, e.g. {'urban': 'strong match'}.",
    )

    # ------------------------------------------------------------------ #
    # Confidence
    # ------------------------------------------------------------------ #
    calibrated_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    uncertainty_bound: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # TODO: use ARRAY(String) with PostgreSQL — store as JSON string for SQLite compat
    confidence_flags_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON array of confidence flag strings.",
    )

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #
    attention_alignment: Mapped[float | None] = mapped_column(Float, nullable=True)
    narrative_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    natural_language: Mapped[str | None] = mapped_column(Text, nullable=True)
    heatmap_diff_s3: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="S3 URI or local path for the diff heatmap PNG.",
    )

    # ------------------------------------------------------------------ #
    # User Feedback
    # ------------------------------------------------------------------ #
    user_feedback: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        doc="Analyst feedback: relevant | not_relevant | partially_relevant.",
    )

    def __repr__(self) -> str:
        return (
            f"<RetrievalResult query={self.query_id} "
            f"rank={self.rank} scene={self.scene_id!r} "
            f"sim={self.cosine_similarity:.3f}>"
        )
