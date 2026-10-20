"""004 — Create scene_explanations table"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scene_explanations",
        sa.Column("scene_id", sa.String(128),
                  sa.ForeignKey("scenes.scene_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("axis_vegetation", sa.Float, nullable=False),
        sa.Column("axis_waterbody", sa.Float, nullable=False),
        sa.Column("axis_urban", sa.Float, nullable=False),
        sa.Column("axis_road", sa.Float, nullable=False),
        sa.Column("ndvi_mean", sa.Float, nullable=True),
        sa.Column("ndvi_std", sa.Float, nullable=True),
        sa.Column("ndwi_mean", sa.Float, nullable=True),
        sa.Column("ndbi_mean", sa.Float, nullable=True),
        sa.Column("vegetation_profile_json", sa.Text, nullable=True),
        sa.Column("water_profile_json", sa.Text, nullable=True),
        sa.Column("urban_profile_json", sa.Text, nullable=True),
        sa.Column("road_features_json", sa.Text, nullable=True),
        sa.Column("attention_grid_json", sa.Text, nullable=True),
        sa.Column("attention_heatmap_s3", sa.Text, nullable=True),
        sa.Column("region_importance", JSONB, nullable=True),
        sa.Column("embedding_uncertainty", sa.Float, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("compute_duration_ms", sa.Integer, nullable=True),
    )
    op.create_index("ix_scene_explanations_model_version", "scene_explanations", ["model_version"])
    op.create_index("ix_scene_explanations_axis_urban", "scene_explanations", ["axis_urban"])


def downgrade() -> None:
    op.drop_table("scene_explanations")
