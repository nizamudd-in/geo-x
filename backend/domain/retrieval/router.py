"""
GeoSense-X — Modality Router
==============================
Decides which FAISS index(es) to search based on source and target modality.
Encapsulates all routing logic so the pipeline stays clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from services.faiss_service import FAISSService, SearchResult

logger = get_logger(__name__)


class ModalityRouter:
    """Routes FAISS searches to the correct index based on modality pair."""

    def __init__(self, faiss: "FAISSService") -> None:
        """
        Initialise the router.

        Args:
            faiss: Loaded FAISSService singleton.
        """
        self._faiss = faiss

    def route(
        self,
        embedding: "np.ndarray",
        source_modality: str,
        target_modality: str,
        k: int,
        filters: dict | None = None,
    ) -> "list[SearchResult]":
        """
        Route the search to the appropriate FAISS index(es).

        Args:
            embedding: [128] float32 L2-normalised query embedding.
            source_modality: Source image modality.
            target_modality: Target modality ("optical"|"sar"|"multispectral"|"all").
            k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult ordered by similarity descending.
        """
        # TODO: implement routing logic
        # Same-modal case
        # if source_modality == target_modality:
        #     return self._faiss.search(embedding, source_modality, target_modality, k, filters)
        # Cross-modal / all-modal
        # else:
        #     return self._faiss.search(embedding, source_modality, target_modality, k, filters)
        raise NotImplementedError("ModalityRouter.route() not yet implemented")
