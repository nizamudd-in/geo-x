"""
GeoSense-X — Celery Application Factory
=========================================
Creates and configures the Celery application for async task processing.

Task queues:
  embedding  → GPU-bound embedding computation jobs (high priority)
  heatmap    → deferred attention heatmap generation
  ingestion  → bulk scene catalog ingestion

Usage:
    # Start worker:
    celery -A workers.celery_app worker -Q heatmap,embedding --loglevel=info

    # Monitor:
    celery -A workers.celery_app flower
"""

from __future__ import annotations

from celery import Celery

from config import settings


def create_celery_app() -> Celery:
    """
    Create and configure the Celery application instance.

    Returns:
        Configured Celery app ready for task registration.
    """
    app = Celery("geosense_x")

    app.conf.update(
        broker_url=settings.CELERY_BROKER_URL,
        result_backend=settings.CELERY_RESULT_BACKEND,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        result_expires=24 * 3600,  # 24h
        task_routes={
            "workers.embedding_tasks.*": {"queue": "embedding"},
            "workers.heatmap_tasks.*": {"queue": "heatmap"},
            "workers.ingestion_tasks.*": {"queue": "ingestion"},
        },
        # TODO: configure task retry policy
        # task_max_retries=3,
        # task_retry_backoff=True,
        # TODO: add Flower monitoring config
        # flower_port=5555,
    )

    # Auto-discover tasks in workers package
    app.autodiscover_tasks(
        ["workers.embedding_tasks", "workers.heatmap_tasks", "workers.ingestion_tasks"]
    )

    return app


celery_app: Celery = create_celery_app()
