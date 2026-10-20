"""
GeoSense-X — SemanticAxisVector ORM Model
===========================================
Maps to the `semantic_axis_vectors` table.

Stores the learned direction vectors for each semantic axis
(vegetation, waterbody, urban, road_network) per model version.

These vectors are produced by training 4 linear probes on the
shared embedding space (ml/training/axis_probes/).
At inference, they are loaded into memory by AxisScorer
and used for fast semantic axis scoring.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from models.orm.base import Base


class SemanticAxisVector(Base):
    """
    Learned semantic axis direction vector for one axis and model version.

    The axis_vector is a 128-dimensional unit vector pointing in the
    direction of increasing semantic content for that axis in the
    shared embedding space.

    axis_score(embedding) = sigmoid(embedding · axis_vector + bias)
    """

    __tablename__ = "semantic_axis_vectors"

    __table_args__ = (
        UniqueConstraint(
            "model_version",
            "axis_name",
            name="uq_axis_vectors_version_name",
        ),
    )

    # ------------------------------------------------------------------ #
    # Primary Key
    # ------------------------------------------------------------------ #
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Model checkpoint version this axis was trained against.",
    )
    axis_name: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Axis name: vegetation | waterbody | urban | road_network.",
    )

    # ------------------------------------------------------------------ #
    # Vector Data
    # ------------------------------------------------------------------ #
    # TODO: replace with pgvector Vector(128) when available
    # axis_vector: Mapped[list[float]] = mapped_column(Vector(128), nullable=False)
    axis_vector_json: Mapped[str] = mapped_column(
        String(8192),  # 128 floats × ~6 chars each = ~768 chars, padded
        nullable=False,
        doc="JSON-serialised 128-d axis direction vector.",
    )
    bias: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Bias term from linear probe: score = sigmoid(z·w + bias).",
    )

    # ------------------------------------------------------------------ #
    # Calibration Parameters
    # ------------------------------------------------------------------ #
    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Similarity score at which P(relevant) = 0.5.",
    )
    temperature: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        doc="Temperature scaling parameter for Platt calibration.",
    )

    # ------------------------------------------------------------------ #
    # Probe Quality Metrics
    # ------------------------------------------------------------------ #
    probe_train_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    probe_val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ------------------------------------------------------------------ #
    # Timestamps
    # ------------------------------------------------------------------ #
    probe_trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<SemanticAxisVector axis={self.axis_name!r} "
            f"version={self.model_version!r} "
            f"val_acc={self.probe_val_accuracy}>"
        )
