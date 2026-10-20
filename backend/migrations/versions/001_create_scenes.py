"""
001 — Create scenes table
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "scenes",
        sa.Column("scene_id", sa.String(128), primary_key=True),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lon", sa.Float, nullable=False),
        sa.Column("scene_date", sa.Date, nullable=True),
        sa.Column("season", sa.String(10), nullable=True),
        sa.Column("lclu_class", sa.String(64), nullable=False),
        sa.Column("lclu_class_id", sa.SmallInteger, nullable=False),
        sa.Column("has_optical", sa.Boolean, default=False, nullable=False),
        sa.Column("has_sar", sa.Boolean, default=False, nullable=False),
        sa.Column("has_multispectral", sa.Boolean, default=False, nullable=False),
        sa.Column("cloud_cover_pct", sa.Float, nullable=True),
        sa.Column("thumbnail_path", sa.Text, nullable=True),
        sa.Column("hdf5_path", sa.Text, nullable=False),
        sa.Column("hdf5_index", sa.Integer, nullable=False),
        sa.Column("source_dataset", sa.String(64), server_default="sen12ms", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("ix_scenes_lclu_class_id", "scenes", ["lclu_class_id"])
    op.create_index("ix_scenes_scene_date", "scenes", ["scene_date"])

    # TODO: add PostGIS geometry column once geoalchemy2 is installed
    # op.add_column("scenes", sa.Column("geometry", Geometry("POINT", srid=4326)))
    # op.create_geospatial_index("ix_scenes_geometry", "scenes", ["geometry"], postgresql_using="gist")


def downgrade() -> None:
    op.drop_table("scenes")
