"""
GeoSense-X — Async Database Engine & Session Factory
=====================================================
Configures SQLAlchemy async engine and provides session management
for dependency injection into FastAPI route handlers.

All database access goes through the session yielded by get_db().
Never create sessions manually in route handlers or services.

Usage:
    # In a route handler (via dependency injection):
    async def my_route(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Scene))

    # In lifespan (startup):
    await init_db()
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Module-level engine and session factory — created once at startup
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _create_engine() -> AsyncEngine:
    """
    Create the SQLAlchemy async engine with connection pool configuration.

    Engine is created once and reused for the application lifetime.
    Pool pre-ping is enabled to detect stale connections.
    """
    return create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        # TODO: add pool_recycle for long-running servers
        # pool_recycle=3600,
    )


def _create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create the async session factory bound to the given engine.

    expire_on_commit=False prevents lazy-loading errors after commit
    in async contexts.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async def init_db() -> None:
    """
    Initialise the database engine and session factory.

    Called once during application startup (core/events.py).
    Verifies the connection is reachable before marking the app as ready.

    Raises:
        sqlalchemy.exc.OperationalError: If the database is unreachable.
    """
    global _engine, _async_session_factory

    logger.info("Initialising database connection", url=settings.DATABASE_URL)

    _engine = _create_engine()
    _async_session_factory = _create_session_factory(_engine)

    # Verify connectivity
    async with _engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))

    logger.info("Database connection established")

    # TODO: run Alembic migrations programmatically in development
    # from alembic.config import Config
    # from alembic import command
    # alembic_cfg = Config("alembic.ini")
    # command.upgrade(alembic_cfg, "head")


async def close_db() -> None:
    """
    Dispose of the database engine connection pool.

    Called during application shutdown (core/events.py).
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database connection pool disposed")
        _engine = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.

    Commits on success, rolls back on any exception.
    Session is closed after the request completes regardless of outcome.

    Yields:
        AsyncSession: A SQLAlchemy async session for this request.

    Raises:
        RuntimeError: If called before init_db().
    """
    if _async_session_factory is None:
        # TODO: raise proper service unavailable error
        raise RuntimeError("Database not initialised. Call init_db() at startup.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def health_check() -> bool:
    """
    Check whether the database connection is healthy.

    Used by GET /ready endpoint to determine readiness.

    Returns:
        True if database is reachable, False otherwise.
    """
    if _engine is None:
        return False
    try:
        async with _engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed", error=str(exc))
        return False


def get_engine() -> AsyncEngine:
    """
    Return the current engine instance.

    Used by Alembic migrations and admin tooling.

    Raises:
        RuntimeError: If called before init_db().
    """
    if _engine is None:
        raise RuntimeError("Database not initialised. Call init_db() at startup.")
    return _engine
