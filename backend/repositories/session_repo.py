"""
GeoSense-X — Retrieval Session Repository
==========================================
All database queries for the `retrieval_sessions` table.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from models.orm.session import RetrievalSession

logger = get_logger(__name__)


class SessionRepository:
    """Data access layer for the retrieval_sessions table."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save_session(self, session: RetrievalSession) -> uuid.UUID:
        """
        Persist a new retrieval session record.

        Args:
            session: Populated RetrievalSession ORM instance.

        Returns:
            The assigned query_id UUID.
        """
        # TODO: implement insert
        # self._db.add(session)
        # await self._db.flush()
        # return session.query_id
        raise NotImplementedError

    async def get_session(self, query_id: uuid.UUID) -> RetrievalSession | None:
        """Retrieve a session by query_id."""
        # TODO: implement select
        raise NotImplementedError

    async def list_sessions(
        self,
        user_id: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> list[RetrievalSession]:
        """List retrieval sessions with optional user filter and pagination."""
        # TODO: implement paginated select
        raise NotImplementedError
