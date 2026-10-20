"""
GeoSense-X — RetrievalSession ORM Model
=========================================
Maps to the `retrieval_sessions` table.

Logs every retrieval request for analytics, replay, and user history.
One row per POST /retrieve/* request.
The query_embedding column enables future session replay without
re-encoding the original image.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.orm.base import Base, TimestampMixin


class RetrievalSession(TimestampMixin, Base):
    """
    Log entry for a single retrieval request.

    Captures the query parameters, computed embedding, axis profile,
    performance metrics, and configuration used for the request.
    """

    __tablename__ = "retrieval_sessions"

    # ------------------------------------------------------------------ #
    # Primary Key
    # ------------------------------------------------------------------ #
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier for this retrieval session.",
    )

    # ------------------------------------------------------------------ #
    # User Context
    # ------------------------------------------------------------------ #
    user_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Authenticated user ID, or null for anonymous requests.",
    )
    api_key_prefix: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        doc="First 8 chars of the API key for audit purposes.",
    )

    # ------------------------------------------------------------------ #
    # Query Configuration
    # ------------------------------------------------------------------ #
    source_modality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Modality of the uploaded query image.",
    )
    target_modality: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Target retrieval modality.",
    )
    top_k: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=10,
        doc="Number of results requested.",
    )
    explanation_depth: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="fast",
        doc="Explanation depth: fast | full.",
    )
    filters: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="Applied search filters (bbox, date range, season) as JSON.",
    )

    # ------------------------------------------------------------------ #
    # Query Embedding & Axis Scores
    # ------------------------------------------------------------------ #
    # TODO: replace Text with pgvector Vector(128)
    # query_embedding: Mapped[list[float]] = mapped_column(Vector(128), nullable=False)
    query_embedding_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Serialised 128-d query embedding (JSON array, temporary).",
    )

    query_axis_vegetation: Mapped[float | None] = mapped_column(Float, nullable=True)
    query_axis_waterbody: Mapped[float | None] = mapped_column(Float, nullable=True)
    query_axis_urban: Mapped[float | None] = mapped_column(Float, nullable=True)
    query_axis_road: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ------------------------------------------------------------------ #
    # Performance
    # ------------------------------------------------------------------ #
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Model version used for this session.",
    )
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    encode_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    explain_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    # TODO: add relationship once RetrievalResult model is created
    # results: Mapped[list["RetrievalResult"]] = relationship(
    #     "RetrievalResult", back_populates="session", lazy="selectin"
    # )

    def __repr__(self) -> str:
        return (
            f"<RetrievalSession id={self.query_id} "
            f"{self.source_modality}→{self.target_modality}>"
        )
