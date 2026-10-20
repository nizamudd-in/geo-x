"""
GeoSense-X — Scene Ingestion Tasks
=====================================
Celery tasks for ingesting new satellite scenes into the catalog.
Each ingestion task: parse metadata → save to DB → compute thumbnail
                     → compute embedding → compute scene explanation.
"""

from __future__ import annotations

from celery import shared_task

from core.logging import get_logger

logger = get_logger(__name__)


@shared_task(
    name="workers.ingestion_tasks.ingest_scene",
    queue="ingestion",
    max_retries=3,
    default_retry_delay=30,
)
def ingest_scene(
    scene_path: str,
    modality: str,
    metadata: dict | None = None,
) -> str:
    """
    Ingest a single satellite scene into the GeoSense-X catalog.

    Steps:
      1. Parse scene metadata (from GeoTIFF tags or provided dict)
      2. Save scene record to `scenes` table
      3. Generate and save 128×128 JPEG thumbnail
      4. Compute embedding → save to `embeddings` table
      5. Compute scene explanation signals → save to `scene_explanations` table

    Args:
        scene_path: Path to the source GeoTIFF or HDF5 file.
        modality: Image modality ("optical", "sar", "multispectral").
        metadata: Optional pre-parsed metadata dict (skips parsing step).

    Returns:
        The scene_id string for the ingested scene.
    """
    # TODO: implement scene ingestion pipeline
    raise NotImplementedError("ingest_scene not yet implemented")


@shared_task(
    name="workers.ingestion_tasks.ingest_bulk",
    queue="ingestion",
)
def ingest_bulk(
    scene_paths: list[str],
    modality: str,
) -> dict:
    """
    Ingest a batch of scenes by queuing individual ingest_scene tasks.

    Args:
        scene_paths: List of file paths to ingest.
        modality: Target modality for all scenes.

    Returns:
        Dict with task IDs: {queued: int, task_ids: list[str]}.
    """
    # TODO: implement bulk ingestion by dispatch
    # task_ids = []
    # for path in scene_paths:
    #     task = ingest_scene.delay(path, modality)
    #     task_ids.append(task.id)
    # return {"queued": len(task_ids), "task_ids": task_ids}
    raise NotImplementedError("ingest_bulk not yet implemented")
