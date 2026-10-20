"""002 — Create embeddings table"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "embeddings",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("scene_id", sa.String(128),
                  sa.ForeignKey("scenes.scene_id", ondelete="CASCADE"), nullable=False),
        sa.Column("modality", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        # TODO: change to vector(128) once pgvector is available
        sa.Column("embedding_json", sa.Text, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("compute_ms", sa.Integer, nullable=True),
        sa.UniqueConstraint("scene_id", "modality", "model_version",
                            name="uq_embeddings_scene_modality_version"),
    )
    op.create_index("ix_embeddings_scene_id", "embeddings", ["scene_id"])
    op.create_index("ix_embeddings_model_version", "embeddings", ["model_version", "modality"])


def downgrade() -> None:
    op.drop_table("embeddings")
