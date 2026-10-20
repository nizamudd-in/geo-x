"""005 — Create semantic_axis_vectors and calibration_records tables"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "semantic_axis_vectors",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("axis_name", sa.String(32), nullable=False),
        sa.Column("axis_vector_json", sa.String(8192), nullable=False),
        sa.Column("bias", sa.Float, nullable=False, default=0.0),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("temperature", sa.Float, nullable=False, default=1.0),
        sa.Column("probe_train_accuracy", sa.Float, nullable=True),
        sa.Column("probe_val_accuracy", sa.Float, nullable=True),
        sa.Column("probe_trained_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("model_version", "axis_name", name="uq_axis_vectors_version_name"),
    )
    op.create_index("ix_axis_vectors_model_version", "semantic_axis_vectors", ["model_version"])

    op.create_table(
        "calibration_records",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("lclu_class", sa.String(64), nullable=False),
        sa.Column("temperature", sa.Float, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("ece_before", sa.Float, nullable=True),
        sa.Column("ece_after", sa.Float, nullable=True),
        sa.Column("calibrated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("model_version", "lclu_class", name="uq_calibration_version_class"),
    )


def downgrade() -> None:
    op.drop_table("calibration_records")
    op.drop_table("semantic_axis_vectors")
