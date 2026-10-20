"""
GeoSense-X — Structured Logging Configuration
===============================================
Configures loguru for structured JSON output in production and
human-readable coloured output in development.

Every log entry includes: timestamp, level, module, function, request_id.
Request ID is bound per-request via middleware (injected from X-Request-ID header
or generated if absent).

Usage:
    from core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Scene encoded", scene_id=scene_id, latency_ms=12)
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """
    Configure loguru for the application.

    Removes default handler and adds a new one with JSON formatting
    in production or coloured text formatting in development.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()

    # TODO: detect ENVIRONMENT from settings and switch format
    # For now, use a human-readable format suitable for development
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )

    logger.add(
        sys.stdout,
        format=fmt,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # TODO: add JSON sink for production
    # logger.add(
    #     sys.stdout,
    #     format="{message}",
    #     level=level,
    #     serialize=True,  # outputs JSON
    #     colorize=False,
    # )

    # TODO: add file rotation sink for persistent logs
    # logger.add(
    #     "logs/geosense_{time}.log",
    #     rotation="100 MB",
    #     retention="7 days",
    #     compression="gz",
    #     level=level,
    #     serialize=True,
    # )


def get_logger(name: str) -> Any:
    """
    Return a loguru logger bound with the given module name.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A loguru logger instance bound with the module context.
    """
    return logger.bind(module=name)


def log_retrieval_request(
    query_id: str,
    source_modality: str,
    target_modality: str,
    top_k: int,
    latency_ms: int,
    result_count: int,
) -> None:
    """
    Emit a structured log entry for a completed retrieval request.

    Called by RetrievalPipeline after successful completion.
    Used for analytics and performance monitoring.

    Args:
        query_id: UUID of the retrieval session.
        source_modality: Source image modality.
        target_modality: Target retrieval modality.
        top_k: Number of results requested.
        latency_ms: Total request latency in milliseconds.
        result_count: Actual number of results returned.
    """
    # TODO: add request_id binding once middleware is implemented
    logger.info(
        "retrieval_complete",
        query_id=query_id,
        source_modality=source_modality,
        target_modality=target_modality,
        top_k=top_k,
        latency_ms=latency_ms,
        result_count=result_count,
    )
