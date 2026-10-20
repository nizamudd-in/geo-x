"""
GeoSense-X — Confidence Service
==================================
High-level interface for the confidence scoring subsystem.
Thin wrapper around ConfidencePipeline with cache integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    import torch
    from models.schemas.confidence import CalibrationParams, ConfidenceBundle
    from services.cache_service import CacheService
    from services.faiss_service import SearchResult

logger = get_logger(__name__)


class ConfidenceService:
    """
    Orchestrates confidence scoring for a set of retrieval candidates.

    Loads calibration parameters from DB at startup (cached in Redis for 1h).
    """

    def __init__(self, cache: "CacheService") -> None:
        """
        Initialise the confidence service.

        Args:
            cache: CacheService instance.
        """
        self._cache = cache
        self._calibration_table: dict[str, "CalibrationParams"] = {}
        self._calibration_loaded: bool = False

    async def load_calibration(self, model_version: str) -> None:
        """
        Load temperature scaling parameters from DB into memory.

        Called at startup after DB connection is established.
        Also refreshed every hour via cache TTL expiry.

        Args:
            model_version: Current model version to load calibration for.
        """
        # TODO: implement DB load via calibration_records table
        # from repositories.calibration_repo import CalibrationRepository
        # params = await CalibrationRepository(db).get_all(model_version)
        # self._calibration_table = {p.lclu_class: p for p in params}
        self._calibration_loaded = True
        logger.info("Calibration parameters loaded (placeholder)", model_version=model_version)

    async def score(
        self,
        query_tensor: "torch.Tensor",
        source_modality: str,
        candidates: "list[SearchResult]",
    ) -> "list[ConfidenceBundle]":
        """
        Compute confidence bundles for all retrieval candidates.

        Args:
            query_tensor: [1, 3, 224, 224] preprocessed query tensor.
            source_modality: Query image modality.
            candidates: Ranked FAISS search results.

        Returns:
            List of ConfidenceBundle, one per candidate (same order).
        """
        # TODO: implement confidence pipeline call
        # from domain.confidence.pipeline import ConfidencePipeline
        # pipeline = ConfidencePipeline(self._calibration_table)
        # return await pipeline.score(query_tensor, source_modality, candidates)
        raise NotImplementedError("ConfidenceService.score() not yet implemented")

    async def get_calibration_params(self, lclu_class: str) -> "CalibrationParams | None":
        """
        Get calibration parameters for a specific LCLU class.

        Args:
            lclu_class: Land cover class name.

        Returns:
            CalibrationParams or None if not found (use global fallback).
        """
        return self._calibration_table.get(lclu_class)

    async def refresh_calibration(self, model_version: str) -> None:
        """
        Force-refresh calibration parameters from DB.

        Called by POST /admin/cache/clear with prefix="calib".

        Args:
            model_version: Model version to refresh for.
        """
        await self._cache.delete_pattern(f"calib:{model_version}")
        await self.load_calibration(model_version)
        logger.info("Calibration refreshed", model_version=model_version)
