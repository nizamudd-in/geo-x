"""
GeoSense-X — Result Composition Service
=========================================
Composes the final RetrievalResponse from raw pipeline outputs.

Responsibilities:
  - Load scene metadata from in-memory dict (loaded at startup)
  - Build thumbnail URLs from scene_id template
  - Merge ranked candidates + explanations + confidence bundles
  - Assign final rank numbers
  - Return fully populated RetrievalResponse
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger
from models.schemas.retrieval import RetrievalResponse, SceneMatch
from models.schemas.scene import SceneMetadata

if TYPE_CHECKING:
    from models.schemas.confidence import ConfidenceBundle
    from models.schemas.explanation import ExplanationBundle
    from services.faiss_service import SearchResult

logger = get_logger(__name__)


class ResultService:
    """
    Composes the final API response from raw retrieval outputs.

    Loaded once at startup and stored in app.state.result_svc.
    Holds an in-memory dict of scene metadata for O(1) lookup.
    """

    def __init__(self) -> None:
        """Initialise the service (call load_metadata() before use)."""
        self._metadata: dict[str, dict] = {}
        self._is_loaded: bool = False

    async def load_metadata(self) -> None:
        """
        Load scene metadata JSON into memory.

        Called once at startup. Metadata JSON is a flat list of scene dicts
        produced by ml/scripts/compute_embeddings.py.

        Raises:
            FileNotFoundError: If metadata.json does not exist.
        """
        path = settings.metadata_path
        if not path.exists():
            logger.warning("Metadata file not found", path=str(path))
            return

        with open(path) as f:
            scenes = json.load(f)

        self._metadata = {scene["scene_id"]: scene for scene in scenes}
        self._is_loaded = True
        logger.info("Scene metadata loaded", count=len(self._metadata))

    def compose(
        self,
        query_id: uuid.UUID,
        source_modality: str,
        target_modality: str,
        candidates: "list[SearchResult]",
        explanations: "list[ExplanationBundle | None]",
        confidence_bundles: "list[ConfidenceBundle | None]",
        latency_ms: int,
        encode_latency_ms: int | None = None,
        search_latency_ms: int | None = None,
        explain_latency_ms: int | None = None,
    ) -> RetrievalResponse:
        """
        Build the final RetrievalResponse from pipeline outputs.

        Args:
            query_id: UUID of the retrieval session.
            source_modality: Source image modality.
            target_modality: Target retrieval modality.
            candidates: Ranked list of FAISS search results.
            explanations: Per-candidate ExplanationBundle list (may contain None).
            confidence_bundles: Per-candidate ConfidenceBundle list (may contain None).
            latency_ms: Total pipeline latency.
            encode_latency_ms: Encoding step latency.
            search_latency_ms: FAISS search step latency.
            explain_latency_ms: Explanation step latency.

        Returns:
            Fully populated RetrievalResponse.
        """
        # TODO: implement composition logic
        results: list[SceneMatch] = []

        for rank, (candidate, explanation, confidence) in enumerate(
            zip(candidates, explanations, confidence_bundles), start=1
        ):
            metadata = self._get_scene_metadata(candidate.scene_id)
            thumbnail_url = self._build_thumbnail_url(candidate.scene_id)

            # TODO: build SceneMatch once all schemas are finalised
            # results.append(SceneMatch(
            #     rank=rank,
            #     scene_id=candidate.scene_id,
            #     modality=candidate.modality,
            #     similarity=candidate.similarity,
            #     thumbnail_url=thumbnail_url,
            #     metadata=metadata,
            #     explanation=explanation,
            #     confidence=confidence,
            # ))

        return RetrievalResponse(
            query_id=query_id,
            source_modality=source_modality,
            target_modality=target_modality,
            top_k=len(results),
            latency_ms=latency_ms,
            encode_latency_ms=encode_latency_ms,
            search_latency_ms=search_latency_ms,
            explain_latency_ms=explain_latency_ms,
            results=results,
        )

    def _get_scene_metadata(self, scene_id: str) -> SceneMetadata:
        """
        Look up scene metadata from the in-memory dict.

        Args:
            scene_id: Scene identifier to look up.

        Returns:
            SceneMetadata populated from the metadata JSON.
            Returns a placeholder if scene not found.
        """
        raw = self._metadata.get(scene_id, {})
        # TODO: map raw dict fields to SceneMetadata schema
        return SceneMetadata(
            lat=raw.get("lat", 0.0),
            lon=raw.get("lon", 0.0),
            scene_date=raw.get("date"),
            season=raw.get("season"),
            lclu_class=raw.get("lclu_class", "unknown"),
        )

    def _build_thumbnail_url(self, scene_id: str) -> str:
        """
        Build the thumbnail URL for a scene.

        Args:
            scene_id: Scene identifier.

        Returns:
            Static file URL string.
        """
        return f"/static/thumbnails/{scene_id}.jpg"
