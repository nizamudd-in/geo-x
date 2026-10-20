"""
GeoSense-X — Retrieval Result Repository
==========================================
All database queries for the `retrieval_results` table.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from models.orm.result import RetrievalResult

logger = get_logger(__name__)


class ResultRepository:
    """Data access layer for the retrieval_results table."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save_results(
        self, query_id: uuid.UUID, results: list[RetrievalResult]
    ) -> None:
        """
        Bulk insert all results for a retrieval session.

        Args:
            query_id: The session UUID these results belong to.
            results: List of populated RetrievalResult ORM instances.
        """
        # TODO: implement bulk insert
        # self._db.add_all(results)
        # await self._db.flush()
        raise NotImplementedError

    async def get_results(self, query_id: uuid.UUID) -> list[RetrievalResult]:
        """Retrieve all results for a session, ordered by rank."""
        # TODO: implement ordered select
        raise NotImplementedError

    async def update_feedback(
        self, result_id: int, feedback: str
    ) -> None:
        """Record user feedback on a specific result."""
        # TODO: implement update
        raise NotImplementedError
