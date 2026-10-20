"""
GeoSense-X — Similarity Gap Scorer
=====================================
Measures the margin between top-1 and each result's similarity.
Large gap = top result is clearly best; small gap = ambiguous ranking.
"""

from __future__ import annotations

import numpy as np


class GapScorer:
    """Computes per-result gap scores from a list of similarity scores."""

    @staticmethod
    def compute_all(similarities: list[float]) -> list[float]:
        """
        Compute gap scores for all results in a session.

        Args:
            similarities: List of cosine similarities in descending order.

        Returns:
            List of gap scores in [0, 1], same order as input.
            gap_score = 1 - (top1_sim - result_sim) / top1_sim
            Value of 1.0 for rank-1 (no gap by definition).
        """
        if not similarities:
            return []

        top1 = similarities[0]
        gap_scores = []
        for sim in similarities:
            if top1 < 1e-8:
                gap_scores.append(0.5)
            else:
                gap = (top1 - sim) / (top1 + 1e-8)
                gap_scores.append(max(0.0, 1.0 - gap))

        return gap_scores
