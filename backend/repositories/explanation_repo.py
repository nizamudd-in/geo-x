"""
GeoSense-X — Scene Explanation Repository
==========================================
All database queries for the `scene_explanations` table.
This is the Level 2 (cold) cache for precomputed per-scene signals.
"""

from __future__ import annotations

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from models.orm.explanation import SceneExplanation

logger = get_logger(__name__)


class ExplanationRepository:
    """Data access layer for the scene_explanations table."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_scene_explanation(
        self, scene_id: str, model_version: str
    ) -> SceneExplanation | None:
        """
        Retrieve precomputed explanation signals for a scene.

        Args:
            scene_id: Scene identifier.
            model_version: Expected model version (returns None if stale).

        Returns:
            SceneExplanation ORM instance or None.
        """
        # TODO: implement select
        raise NotImplementedError

    async def save_scene_explanation(self, explanation: SceneExplanation) -> None:
        """Persist or update a scene explanation record."""
        # TODO: implement upsert (merge/on_conflict)
        raise NotImplementedError

    async def batch_get_explanations(
        self, scene_ids: list[str], model_version: str
    ) -> dict[str, SceneExplanation]:
        """Retrieve explanations for multiple scenes at once."""
        # TODO: implement batch select with IN clause
        raise NotImplementedError

    async def mark_all_stale(self, old_model_version: str) -> int:
        """
        Mark all scene explanations from old model as stale.
        Called when a new model version is deployed.
        Returns count of rows affected.
        """
        # TODO: implement bulk update
        raise NotImplementedError
