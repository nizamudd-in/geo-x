"""
GeoSense-X — Narrative Categorizer
======================================
Classifies (query, result) explanation into one of four narrative categories.
Categories drive template selection in the narrative generator.
"""

from __future__ import annotations

from models.schemas.explanation import NarrativeCategoryEnum, SemanticAxisBundle


class NarrativeCategorizer:
    """Assigns a narrative category based on similarity and axis divergence."""

    # Thresholds
    HIGH_SIM_THRESHOLD = 0.80
    MEDIUM_SIM_THRESHOLD = 0.60
    LOW_CONFIDENCE_THRESHOLD = 0.50

    @classmethod
    def categorize(
        cls,
        global_similarity: float,
        axis_bundle: SemanticAxisBundle,
        confidence_score: float,
    ) -> NarrativeCategoryEnum:
        """
        Assign narrative category based on similarity and explanation signals.

        Args:
            global_similarity: Raw cosine similarity from FAISS [0, 1].
            axis_bundle: Pairwise axis comparison bundle.
            confidence_score: Composite confidence score [0, 1].

        Returns:
            NarrativeCategoryEnum value.
        """
        # CASE D: Low confidence → uncertain regardless of similarity
        if confidence_score < cls.LOW_CONFIDENCE_THRESHOLD:
            return NarrativeCategoryEnum.UNCERTAIN

        divergent = axis_bundle.divergent_axes

        # CASE A: High similarity, no divergent axes
        if global_similarity >= cls.HIGH_SIM_THRESHOLD and not divergent:
            return NarrativeCategoryEnum.STRONG_MATCH

        # CASE B: High similarity, some divergent axes
        if global_similarity >= cls.HIGH_SIM_THRESHOLD and divergent:
            return NarrativeCategoryEnum.PARTIAL_MATCH

        # CASE C: Moderate similarity
        if global_similarity >= cls.MEDIUM_SIM_THRESHOLD:
            return NarrativeCategoryEnum.MODERATE_MATCH

        # Default: uncertain match
        return NarrativeCategoryEnum.UNCERTAIN
