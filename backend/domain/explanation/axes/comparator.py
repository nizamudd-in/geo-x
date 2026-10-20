"""
GeoSense-X — Semantic Axis Comparator
========================================
Computes pairwise axis similarity between a (query, result) embedding pair.
Produces the SemanticAxisBundle included in every ExplanationBundle.
"""

from __future__ import annotations

import numpy as np

from domain.explanation.axes.scorer import AXIS_NAMES, AxisScorer
from models.schemas.explanation import AxisScore, AxisVerdictEnum, SemanticAxisBundle


class AxisComparator:
    """Computes pairwise axis comparisons between query and result embeddings."""

    @staticmethod
    def compare(
        query_embedding: np.ndarray,
        result_embedding: np.ndarray,
    ) -> SemanticAxisBundle:
        """
        Build a SemanticAxisBundle comparing query and result.

        Args:
            query_embedding: [128] L2-normalised query embedding.
            result_embedding: [128] L2-normalised result embedding.

        Returns:
            SemanticAxisBundle with per-axis scores, similarities, and verdicts.
        """
        q_scores = AxisScorer.score(query_embedding)
        r_scores = AxisScorer.score(result_embedding)

        axis_scores: dict[str, AxisScore] = {}
        for axis in AXIS_NAMES:
            q = q_scores[axis]
            r = r_scores[axis]
            sim = 1.0 - abs(q - r)

            # Determine verdict
            if sim > 0.75:
                verdict = AxisVerdictEnum.STRONG_MATCH
            elif sim > 0.40:
                verdict = AxisVerdictEnum.PARTIAL_MATCH
            else:
                verdict = AxisVerdictEnum.SIGNIFICANT_DIVERGENCE

            # Determine divergence direction
            divergence_direction: str | None = None
            if sim < 0.40:
                if q > r:
                    divergence_direction = f"query has more {axis.replace('_', ' ')}"
                else:
                    divergence_direction = f"result has more {axis.replace('_', ' ')}"

            axis_scores[axis] = AxisScore(
                query_score=round(q, 4),
                result_score=round(r, 4),
                similarity=round(sim, 4),
                verdict=verdict,
                divergence_direction=divergence_direction,
            )

        # TODO: SemanticAxisBundle field names use underscores — map AXIS_NAMES accordingly
        return SemanticAxisBundle(
            vegetation=axis_scores.get("vegetation", _neutral_axis("vegetation")),
            waterbody=axis_scores.get("waterbody", _neutral_axis("waterbody")),
            urban=axis_scores.get("urban", _neutral_axis("urban")),
            road_network=axis_scores.get("road_network", _neutral_axis("road_network")),
        )


def _neutral_axis(name: str) -> AxisScore:
    """Return a neutral AxisScore when data is unavailable."""
    return AxisScore(
        query_score=0.5,
        result_score=0.5,
        similarity=1.0,
        verdict=AxisVerdictEnum.STRONG_MATCH,
        divergence_direction=None,
    )
