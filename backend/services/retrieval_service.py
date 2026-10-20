"""
GeoSense-X — Retrieval Service
================================
Orchestrates the full Image → Embedding → FAISS → Response pipeline.
Single public method: retrieve().
"""
from __future__ import annotations

import logging
import time
import uuid

from config import settings
from schemas.retrieval import RetrievalResponse, SceneMatch
from services.encoder_service import EncoderService
from services.faiss_service import FAISSService

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Thin orchestration layer between the API and the ML services.
    Holds references to the encoder and FAISS singletons loaded at startup.
    """

    def __init__(self, encoder: EncoderService, faiss: FAISSService) -> None:
        self.encoder = encoder
        self.faiss = faiss

    def retrieve(
        self,
        image_bytes: bytes,
        source_modality: str,
        target_modality: str,
        k: int | None = None,
    ) -> RetrievalResponse:
        """
        Full retrieval pipeline for one query image.

        Args:
            image_bytes:      Raw bytes of the uploaded image.
            source_modality:  Modality of the query image ("optical" | "sar").
            target_modality:  Target retrieval modality ("optical" | "sar" | "all").
            k:                Number of results. Defaults to settings.DEFAULT_TOP_K.

        Returns:
            RetrievalResponse with top-K ranked SceneMatch objects.
        """
        k = min(k or settings.DEFAULT_TOP_K, settings.MAX_TOP_K)
        query_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        # 1. Encode query image into the shared embedding space
        embedding = self.encoder.encode(image_bytes, source_modality)
        t_encode = time.perf_counter()
        logger.debug(
            "Encoded  |  modality=%s  |  %.1f ms",
            source_modality,
            (t_encode - t0) * 1000,
        )

        # 2. FAISS similarity search
        matches = self.faiss.search(embedding, source_modality, target_modality, k)
        t_search = time.perf_counter()
        logger.debug(
            "Searched |  target=%s  |  hits=%d  |  %.1f ms",
            target_modality,
            len(matches),
            (t_search - t_encode) * 1000,
        )

        # 3. Build response
        results = [
            SceneMatch(
                rank=i + 1,
                scene_id=m.scene_id,
                modality=m.modality,
                similarity=round(m.similarity, 6),
                thumbnail_url=f"/static/thumbnails/{m.scene_id}.jpg",
                lat=m.metadata.get("lat"),
                lon=m.metadata.get("lon"),
                lclu_class=m.metadata.get("lclu_class"),
                season=m.metadata.get("season"),
                scene_date=m.metadata.get("date") or m.metadata.get("scene_date"),
            )
            for i, m in enumerate(matches)
        ]

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "retrieve  |  query_id=%s  |  %s→%s  |  %d results  |  %d ms",
            query_id,
            source_modality,
            target_modality,
            len(results),
            latency_ms,
        )

        return RetrievalResponse(
            query_id=query_id,
            source_modality=source_modality,
            target_modality=target_modality,
            latency_ms=latency_ms,
            result_count=len(results),
            results=results,
        )
