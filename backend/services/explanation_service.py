"""
GeoSense-X — Explanation Service
===================================
High-level interface for the explanation subsystem.

Thin orchestration layer that:
  1. Checks cache before invoking the pipeline
  2. Calls ExplanationPipeline for actual computation
  3. Writes results to cache
  4. Dispatches async Celery tasks for deferred heatmap generation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    import torch
    from models.schemas.explanation import ExplanationBundle
    from services.cache_service import CacheService
    from services.faiss_service import SearchResult

logger = get_logger(__name__)


class ExplanationService:
    """
    Orchestrates the Explainable Retrieval Engine (XRE) for a result set.

    Checks Redis and DB caches before running the explanation pipeline.
    Dispatches async heatmap generation after returning fast-mode results.
    """

    def __init__(self, cache: "CacheService") -> None:
        """
        Initialise the explanation service.

        Args:
            cache: CacheService instance for Redis + DB cache access.
        """
        self._cache = cache

    async def explain(
        self,
        query_id: str,
        query_embedding: "np.ndarray",
        query_tensor: "torch.Tensor | None",
        candidates: "list[SearchResult]",
        depth: str = "fast",
    ) -> "list[ExplanationBundle]":
        """
        Compute or retrieve explanations for all retrieval candidates.

        Args:
            query_id: UUID string of the retrieval session.
            query_embedding: [128] float32 query embedding.
            query_tensor: [1, 3, 224, 224] preprocessed tensor (needed for
                          "full" mode attention extraction; may be None for "fast").
            candidates: Ranked list of FAISS search results to explain.
            depth: "fast" (axes + confidence + narrative) or
                   "full" (+ attention heatmaps).

        Returns:
            List of ExplanationBundle, one per candidate (same order as candidates).
        """
        bundles = []

        for candidate in candidates:
            scene_id = candidate.scene_id

            # Check hot cache first
            cached = await self._cache.get_explanation(query_id, scene_id)
            if cached:
                logger.debug("Explanation cache hit", scene_id=scene_id)
                # TODO: deserialise from dict to ExplanationBundle
                # bundles.append(ExplanationBundle(**cached))
                continue

            # TODO: check scene_explanations DB table for precomputed signals
            # scene_expl = await self._cache.get_scene_explanation(scene_id)
            # if not scene_expl:
            #     scene_expl = await explanation_repo.get_scene_explanation(scene_id, model_version)
            #     if scene_expl:
            #         await self._cache.set_scene_explanation(scene_id, scene_expl)

            # TODO: call ExplanationPipeline
            # from domain.explanation.pipeline import ExplanationPipeline
            # pipeline = ExplanationPipeline()
            # bundle = await pipeline.explain_pair(
            #     query_embedding=query_embedding,
            #     query_tensor=query_tensor,
            #     candidate=candidate,
            #     scene_signals=scene_expl,
            #     depth=depth,
            # )

            # TODO: cache result
            # await self._cache.set_explanation(query_id, scene_id, bundle.model_dump())

            # TODO: dispatch async heatmap task if not already computed
            # from workers.heatmap_tasks import generate_heatmap_pair
            # generate_heatmap_pair.delay(query_id, scene_id, query_bytes)

        # TODO: return bundles
        raise NotImplementedError("ExplanationService.explain() not yet implemented")

    async def get_heatmap_png(
        self, query_id: str, scene_id: str, view: str
    ) -> bytes | None:
        """
        Retrieve a cached heatmap PNG.

        Args:
            query_id: Session UUID string.
            scene_id: Scene identifier.
            view: "query" | "result" | "diff".

        Returns:
            PNG bytes or None if not yet computed.
        """
        return await self._cache.get_heatmap(query_id, scene_id, view)

    async def trigger_heatmap_async(
        self, query_id: str, scene_id: str, query_bytes: bytes
    ) -> str:
        """
        Dispatch async Celery task for heatmap generation.

        Args:
            query_id: Session UUID string.
            scene_id: Scene identifier.
            query_bytes: Raw query image bytes for re-encoding.

        Returns:
            Celery task ID string.
        """
        # TODO: implement Celery task dispatch
        # from workers.heatmap_tasks import generate_heatmap_pair
        # task = generate_heatmap_pair.delay(query_id, scene_id, query_bytes)
        # return task.id
        raise NotImplementedError("Async heatmap dispatch not yet implemented")
