"""
GeoSense-X — Async Heatmap Generation Tasks
=============================================
Celery tasks for deferred attention heatmap generation.
Dispatched by ExplanationService after returning fast-mode API response.

These tasks run in the "heatmap" queue and may take 200-800ms each.
"""

from __future__ import annotations

from celery import shared_task

from core.logging import get_logger

logger = get_logger(__name__)


@shared_task(
    name="workers.heatmap_tasks.generate_heatmap_pair",
    queue="heatmap",
    max_retries=2,
    default_retry_delay=5,
)
def generate_heatmap_pair(
    query_id: str,
    scene_id: str,
    query_image_bytes: bytes,
    model_version: str | None = None,
) -> dict:
    """
    Generate attention heatmaps for a (query, result) pair.

    This task:
      1. Loads the query image from bytes
      2. Loads the result scene from HDF5
      3. Runs full attention rollout on both (T=20 MC Dropout for query)
      4. Generates query heatmap, result heatmap, diff overlay PNGs
      5. Uploads PNGs to storage backend
      6. Updates Redis cache with heatmap URLs
      7. Updates DB retrieval_results row with heatmap_diff_s3

    Args:
        query_id: Retrieval session UUID string.
        scene_id: Result scene identifier.
        query_image_bytes: Raw bytes of the original query image.
        model_version: Model version to use (uses current if None).

    Returns:
        Dict with heatmap URLs: {query: str, result: str, diff: str}.
    """
    # TODO: implement heatmap generation
    # from domain.explanation.attention.extractor import AttentionExtractor
    # from domain.explanation.attention.rollout import AttentionRollout
    # from domain.explanation.attention.heatmap import HeatmapRenderer
    # from core.storage import get_storage
    # from core.redis_client import get_redis_client
    #
    # Load model (from shared worker state)
    # model = _get_model(model_version)
    #
    # Process query image
    # query_tensor = Preprocessor.preprocess(query_image_bytes, source_modality)
    # query_attn = AttentionExtractor.extract(model, query_tensor, head_aggregation="none")
    # query_rollout = AttentionRollout.compute(all_layer_attentions)
    # query_heatmap_png = HeatmapRenderer(query_rollout).to_png_bytes()
    #
    # Process result scene
    # result_array = load_from_hdf5(scene_id)
    # result_tensor = Preprocessor.preprocess(result_array, result_modality)
    # result_attn = AttentionExtractor.extract(model, result_tensor, head_aggregation="none")
    # result_rollout = AttentionRollout.compute(all_layer_attentions)
    # result_heatmap_png = HeatmapRenderer(result_rollout).to_png_bytes()
    #
    # Generate diff
    # diff_png = HeatmapRenderer.diff_overlay(query_rollout_256, result_rollout_256)
    #
    # Upload to storage
    # storage = get_storage()
    # q_url = await storage.upload(f"heatmaps/{query_id}_{scene_id}_q.png", query_heatmap_png)
    # r_url = await storage.upload(f"heatmaps/{query_id}_{scene_id}_r.png", result_heatmap_png)
    # d_url = await storage.upload(f"heatmaps/{query_id}_{scene_id}_d.png", diff_png)
    #
    # Update Redis cache
    # redis = await get_redis_client()
    # if redis:
    #     await redis.setex(f"heatmap:{query_id}:{scene_id}:q", 7*86400, query_heatmap_png)
    #     await redis.setex(f"heatmap:{query_id}:{scene_id}:r", 7*86400, result_heatmap_png)
    #     await redis.setex(f"heatmap:{query_id}:{scene_id}:d", 7*86400, diff_png)
    #
    # return {"query": q_url, "result": r_url, "diff": d_url}

    logger.info("Heatmap task triggered (placeholder)", query_id=query_id, scene_id=scene_id)
    return {"query": None, "result": None, "diff": None}
