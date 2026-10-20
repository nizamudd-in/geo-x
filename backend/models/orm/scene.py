"""
GeoSense-X — SatelliteScene ORM Model
=======================================
Maps to the `scenes` table — master catalog of all satellite scene patches.

Each row represents one 256×256 patch with known geographic coordinates,
available modalities, and MODIS land cover class label.
The geometry column uses PostGIS for spatial queries (bbox filters).

Relationships:
  - One SatelliteScene → many EmbeddingRecord (one per modality per version)
  - One SatelliteScene → one SceneExplanation (precomputed at ingestion)
"""

from __future__ import annotations

from sqlalchemy import ARRAY, Boolean, Date, Float, Integer, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

# TODO: import geoalchemy2 Geometry once PostGIS extension is confirmed installed
# from geoalchemy2 import Geometry

from models.orm.base import Base, TimestampMixin


class SatelliteScene(TimestampMixin, Base):
    """
    Master catalog entry for a single satellite scene patch.

    A scene is a 256×256 pixel patch extracted from Sentinel-1 or
    Sentinel-2 imagery. Multiple modalities (optical, SAR, MS) may
    share the same scene_id if they are co-registered.
    """

    __tablename__ = "scenes"

    # ------------------------------------------------------------------ #
    # Primary Key
    # ------------------------------------------------------------------ #
    scene_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        doc="Unique scene identifier, e.g. 'ROIs1158_spring_p0001'.",
    )

    # ------------------------------------------------------------------ #
    # Geographic
    # ------------------------------------------------------------------ #
    lat: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="WGS84 latitude of scene centre.",
    )
    lon: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="WGS84 longitude of scene centre.",
    )

    # TODO: replace with PostGIS Geometry column once geoalchemy2 is configured
    # geometry: Mapped[Any] = mapped_column(
    #     Geometry("POINT", srid=4326),
    #     nullable=False,
    #     doc="PostGIS POINT geometry for spatial indexing.",
    # )

    # ------------------------------------------------------------------ #
    # Temporal
    # ------------------------------------------------------------------ #
    scene_date: Mapped[Date | None] = mapped_column(
        Date,
        nullable=True,
        doc="Acquisition date of the scene.",
    )
    season: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        doc="Season of acquisition: spring | summer | fall | winter.",
    )

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    lclu_class: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="MODIS land cover / land use class name.",
    )
    lclu_class_id: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        doc="MODIS land cover class integer ID (0–32).",
    )

    # ------------------------------------------------------------------ #
    # Modality
    # ------------------------------------------------------------------ #
    # TODO: change to PostgreSQL ARRAY once asyncpg is confirmed
    # available_modalities: Mapped[list[str]] = mapped_column(ARRAY(String), ...)
    has_optical: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="True if optical (RGB) imagery is available for this scene.",
    )
    has_sar: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="True if SAR (VV/VH) imagery is available for this scene.",
    )
    has_multispectral: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="True if multispectral (13-band) imagery is available.",
    )

    # ------------------------------------------------------------------ #
    # Quality
    # ------------------------------------------------------------------ #
    cloud_cover_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Percentage cloud cover (0-100). Null for SAR scenes.",
    )

    # ------------------------------------------------------------------ #
    # Storage References
    # ------------------------------------------------------------------ #
    thumbnail_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Local path or S3 URI to the precomputed 128×128 JPEG thumbnail.",
    )
    hdf5_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Path to the HDF5 shard containing raw scene data.",
    )
    hdf5_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Row index within the HDF5 shard for fast random access.",
    )
    source_dataset: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default="sen12ms",
        doc="Dataset of origin, e.g. 'sen12ms', 'ucmerced'.",
    )

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #
    # TODO: add relationships once child models are created
    # embeddings: Mapped[list["EmbeddingRecord"]] = relationship(
    #     "EmbeddingRecord", back_populates="scene", lazy="selectin"
    # )
    # scene_explanation: Mapped["SceneExplanation | None"] = relationship(
    #     "SceneExplanation", back_populates="scene", lazy="selectin"
    # )

    def __repr__(self) -> str:
        return f"<SatelliteScene id={self.scene_id!r} class={self.lclu_class!r}>"
