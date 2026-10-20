"""
GeoSense-X — Explanation Pipeline Orchestrator
================================================
Drives all explanation signals for a set of retrieval candidates.

Two execution modes:
  "fast" — axis scores + narrative + confidence (< 80ms overhead)
  "full"  — + attention rollout + heatmap PNGs (< 500ms overhead)

Dispatches async Celery task for heatmap generation in both modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    import torch
    from models.schemas.explanation import ExplanationBundle
    from services.faiss_service import SearchResult

logger = get_logger(__name__)


class ExplanationPipeline:
    """Orchestrates all explanation signals for a (query, result) pair."""

    async def explain_pair(
        self,
        query_embedding: "np.ndarray",
        query_tensor: "torch.Tensor | None",
        candidate: "SearchResult",
        scene_signals: dict | None,
        depth: str = "fast",
    ) -> "ExplanationBundle":
        """
        Compute a full ExplanationBundle for one (query, result) pair.

        Args:
            query_embedding: [128] L2-normalised query embedding.
            query_tensor: [1, 3, 224, 224] preprocessed tensor (for "full" mode).
            candidate: FAISS search result to explain.
            scene_signals: Precomputed per-scene signals from scene_explanations table.
            depth: "fast" or "full".

        Returns:
            ExplanationBundle with all available signals populated.
        """
        # TODO: Step 1 — Semantic axis scoring
        # from domain.explanation.axes.scorer import AxisScorer
        # from domain.explanation.axes.comparator import AxisComparator
        # query_axes = AxisScorer.score(query_embedding)
        # result_axes = AxisScorer.score(candidate.embedding)
        # axis_bundle = AxisComparator.compare(query_axes, result_axes)

        # TODO: Step 2 — Attention alignment (from cached 18×18 grid)
        # if scene_signals and scene_signals.get("attention_grid_json"):
        #     result_grid = np.array(json.loads(scene_signals["attention_grid_json"]))
        #     from domain.explanation.attention.alignment import AttentionAligner
        #     alignment = AttentionAligner.score(query_attention_grid, result_grid)

        # TODO: Step 3 — Region importance (from cached scene_signals)
        # region_importance = scene_signals.get("region_importance", [])

        # TODO: Step 4 — Narrative generation
        # from domain.explanation.narrative.categorizer import NarrativeCategorizer
        # from domain.explanation.narrative.generator import NarrativeGenerator
        # category = NarrativeCategorizer.categorize(candidate.similarity, axis_bundle, confidence)
        # narrative = NarrativeGenerator.generate(axis_bundle, confidence, region_importance, category)

        # TODO: Step 5 — Full mode: attention extraction and heatmap
        # if depth == "full" and query_tensor is not None:
        #     from domain.explanation.attention.extractor import AttentionExtractor
        #     from domain.explanation.attention.heatmap import HeatmapRenderer
        #     query_attn = AttentionExtractor.extract(model, query_tensor)
        #     result_attn = AttentionExtractor.extract(model, result_tensor)
        #     query_heatmap_png = HeatmapRenderer(query_attn).to_png_bytes()
        #     result_heatmap_png = HeatmapRenderer(result_attn).to_png_bytes()
        #     diff_png = HeatmapRenderer.diff(query_attn, result_attn)

        raise NotImplementedError("ExplanationPipeline.explain_pair() not yet implemented")
