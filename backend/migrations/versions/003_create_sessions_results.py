"""003 — Create retrieval_sessions and retrieval_results tables"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_sessions",
        sa.Column("query_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.String(128), nullable=True),
        sa.Column("api_key_prefix", sa.String(16), nullable=True),
        sa.Column("source_modality", sa.String(32), nullable=False),
        sa.Column("target_modality", sa.String(32), nullable=False),
        sa.Column("top_k", sa.SmallInteger, nullable=False, default=10),
        sa.Column("explanation_depth", sa.String(16), nullable=False, default="fast"),
        sa.Column("filters", JSONB, nullable=True),
        sa.Column("query_embedding_json", sa.Text, nullable=True),
        sa.Column("query_axis_vegetation", sa.Float, nullable=True),
        sa.Column("query_axis_waterbody", sa.Float, nullable=True),
        sa.Column("query_axis_urban", sa.Float, nullable=True),
        sa.Column("query_axis_road", sa.Float, nullable=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("total_latency_ms", sa.Integer, nullable=True),
        sa.Column("encode_latency_ms", sa.Integer, nullable=True),
        sa.Column("search_latency_ms", sa.Integer, nullable=True),
        sa.Column("explain_latency_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_retrieval_sessions_created_at", "retrieval_sessions", ["created_at"])
    op.create_index("ix_retrieval_sessions_user_id", "retrieval_sessions", ["user_id", "created_at"])

    op.create_table(
        "retrieval_results",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("query_id", UUID(as_uuid=True),
                  sa.ForeignKey("retrieval_sessions.query_id", ondelete="CASCADE"), nullable=False),
        sa.Column("scene_id", sa.String(128),
                  sa.ForeignKey("scenes.scene_id"), nullable=False),
        sa.Column("rank", sa.SmallInteger, nullable=False),
        sa.Column("modality", sa.String(32), nullable=False),
        sa.Column("cosine_similarity", sa.Float, nullable=False),
        sa.Column("axis_sim_vegetation", sa.Float, nullable=True),
        sa.Column("axis_sim_waterbody", sa.Float, nullable=True),
        sa.Column("axis_sim_urban", sa.Float, nullable=True),
        sa.Column("axis_sim_road", sa.Float, nullable=True),
        sa.Column("axis_verdicts", JSONB, nullable=True),
        sa.Column("calibrated_prob", sa.Float, nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("confidence_tier", sa.String(16), nullable=True),
        sa.Column("uncertainty_bound", sa.String(16), nullable=True),
        sa.Column("confidence_flags_json", sa.Text, nullable=True),
        sa.Column("attention_alignment", sa.Float, nullable=True),
        sa.Column("narrative_category", sa.String(64), nullable=True),
        sa.Column("natural_language", sa.Text, nullable=True),
        sa.Column("heatmap_diff_s3", sa.Text, nullable=True),
        sa.Column("user_feedback", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("query_id", "rank", name="uq_results_query_rank"),
        sa.UniqueConstraint("query_id", "scene_id", name="uq_results_query_scene"),
    )
    op.create_index("ix_retrieval_results_query_id", "retrieval_results", ["query_id", "rank"])
    op.create_index("ix_retrieval_results_scene_id", "retrieval_results", ["scene_id"])


def downgrade() -> None:
    op.drop_table("retrieval_results")
    op.drop_table("retrieval_sessions")
