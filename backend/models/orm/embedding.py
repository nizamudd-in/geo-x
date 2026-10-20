"""
GeoSense-X — EmbeddingRecord ORM Model
========================================
Maps to the `embeddings` table.

Stores precomputed L2-normalised embedding vectors for each
(scene, modality, model_version) triple. The embedding column
uses pgvector for optional ANN fallback queries.

One scene × one modality × one model version = one row.
When a new model version is deployed, old rows are marked stale
and eventually replaced by the recomputation pipeline.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

# TODO: import pgvector once the extension is confirmed installed
# from pgvector.sqlalchemy import Vector

from models.orm.base import Base


class EmbeddingRecord(Base):
    """
    Persisted embedding vector for a (scene, modality, model_version) triple.

    The embedding field stores a 128-dimensional L2-normalised float32 vector.
    Used as a cache and for pgvector ANN queries (fallback when FAISS is unavailable).
    """

    __tablename__ = "embeddings"

    __table_args__ = (
        UniqueConstraint(
            "scene_id",
            "modality",
            "model_version",
            name="uq_embeddings_scene_modality_version",
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
    # Foreign Key
    # ------------------------------------------------------------------ #
    scene_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("scenes.scene_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Scene this embedding belongs to.",
    )

    # ------------------------------------------------------------------ #
    # Embedding Metadata
    # ------------------------------------------------------------------ #
    modality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Image modality: optical | sar | multispectral.",
    )
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Model checkpoint version string that produced this embedding.",
    )

    # ------------------------------------------------------------------ #
    # Embedding Vector
    # ------------------------------------------------------------------ #
    # TODO: replace with pgvector Vector(128) once extension is available
    # embedding: Mapped[list[float]] = mapped_column(Vector(128), nullable=False)
    # For now, store as JSON array (slower but works without pgvector extension)
    # embedding_json: Mapped[str] = mapped_column(Text, nullable=False)

    # ------------------------------------------------------------------ #
    # Timestamps
    # ------------------------------------------------------------------ #
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="When this embedding was computed.",
    )
    compute_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        doc="Time taken to compute this embedding in milliseconds.",
    )

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    # TODO: add relationship back to SatelliteScene
    # scene: Mapped["SatelliteScene"] = relationship("SatelliteScene", back_populates="embeddings")

    def __repr__(self) -> str:
        return (
            f"<EmbeddingRecord scene={self.scene_id!r} "
            f"modality={self.modality!r} version={self.model_version!r}>"
        )
