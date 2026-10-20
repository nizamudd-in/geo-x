"""
GeoSense-X — Application Lifecycle Event Handlers
===================================================
Centralises all startup and shutdown logic called from main.py's lifespan.

The startup sequence is ordered — each step depends on the previous:
  1. Config validation (Pydantic — already done at import)
  2. Logging
  3. Database
  4. Redis
  5. ML model (EncoderService)
  6. FAISS indexes (FAISSService)
  7. Metadata JSON
  8. Axis vectors (AxisScorer)
  9. Calibration params (CalibrationTable)
  10. Celery connectivity check

Keeping this in a dedicated module makes it easy to mock in tests.
"""

from __future__ import annotations

from fastapi import FastAPI

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)


async def on_startup(app: FastAPI) -> None:
    """
    Run all startup initialisation steps in order.

    Stores singletons on app.state so they are accessible from
    FastAPI dependencies without circular imports.

    Args:
        app: The FastAPI application instance.

    Raises:
        RuntimeError: If any critical step fails (DB, model, indexes).
    """
    logger.info("=== GeoSense-X Startup ===", version=settings.APP_VERSION)

    # Step 1: Database
    # TODO: uncomment when core/database.py is implemented
    # from core.database import init_db
    # await init_db()
    # logger.info("✓ Database connected")

    # Step 2: Redis
    # TODO: uncomment when core/redis_client.py is implemented
    # from core.redis_client import init_redis
    # app.state.redis = await init_redis()
    # logger.info("✓ Redis connected")

    # Step 3: ML Model
    # TODO: uncomment when services/encoder_service.py is implemented
    # from services.encoder_service import EncoderService
    # app.state.encoder = EncoderService()
    # await app.state.encoder.load()
    # logger.info("✓ ML model loaded", model_version=app.state.encoder.model_version)

    # Step 4: FAISS Indexes
    # TODO: uncomment when services/faiss_service.py is implemented
    # from services.faiss_service import FAISSService
    # app.state.faiss = FAISSService()
    # await app.state.faiss.load_indexes()
    # logger.info("✓ FAISS indexes loaded")

    # Step 5: Metadata
    # TODO: uncomment when services/result_service.py is implemented
    # from services.result_service import ResultService
    # app.state.result_svc = ResultService()
    # await app.state.result_svc.load_metadata()
    # logger.info("✓ Scene metadata loaded")

    # Step 6: Axis Vectors (requires DB + model version)
    # TODO: uncomment when domain/explanation/axes/scorer.py is implemented
    # from domain.explanation.axes.scorer import AxisScorer
    # await AxisScorer.load_axis_vectors(model_version=app.state.encoder.model_version)
    # logger.info("✓ Semantic axis vectors loaded")

    # Step 7: Calibration Params
    # TODO: uncomment when domain/confidence/calibration.py is implemented
    # from domain.confidence.calibration import CalibrationTable
    # await CalibrationTable.load(model_version=app.state.encoder.model_version)
    # logger.info("✓ Calibration parameters loaded")

    # Step 8: Celery connectivity check (non-fatal)
    # TODO: uncomment when workers/celery_app.py is implemented
    # try:
    #     from workers.celery_app import celery_app
    #     celery_app.control.ping(timeout=1)
    #     logger.info("✓ Celery workers reachable")
    # except Exception:
    #     logger.warning("⚠ Celery workers not reachable — async tasks disabled")

    logger.info("=== GeoSense-X Ready ===")


async def on_shutdown(app: FastAPI) -> None:
    """
    Run all cleanup steps in reverse startup order.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("=== GeoSense-X Shutdown ===")

    # Step 1: Release GPU memory
    # TODO: uncomment when encoder_service is implemented
    # if hasattr(app.state, "encoder"):
    #     app.state.encoder.release()
    #     logger.info("✓ GPU memory released")

    # Step 2: Close Redis
    # TODO: uncomment when redis_client is implemented
    # from core.redis_client import close_redis
    # await close_redis()
    # logger.info("✓ Redis connection closed")

    # Step 3: Close database
    # TODO: uncomment when database is implemented
    # from core.database import close_db
    # await close_db()
    # logger.info("✓ Database connection closed")

    logger.info("=== GeoSense-X Stopped ===")
