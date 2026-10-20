"""
GeoSense-X — Composite Confidence Scorer
==========================================
Combines calibrated probability, uncertainty, gap, and modality consistency
into a single confidence score with tier and flags.
"""

from __future__ import annotations

from config import settings
from models.schemas.confidence import ConfidenceBundle
from models.schemas.common import ConfidenceTierEnum


class CompositeScorer:
    """Computes the weighted composite confidence score and assigns tier/flags."""

    # Uncertainty normalisation: map [0, max_uncertainty] → [0, 1]
    _MAX_UNCERTAINTY = 0.15

    @staticmethod
    def compute(
        calibrated_prob: float,
        uncertainty_scalar: float,
        gap_score: float,
        raw_similarity: float,
        modality_consistency: float = 1.0,
        source_modality: str = "optical",
        target_modality: str = "optical",
        query_season: str | None = None,
        result_season: str | None = None,
        query_axis_scores: dict | None = None,
    ) -> ConfidenceBundle:
        """
        Compute composite confidence score and assemble ConfidenceBundle.

        Args:
            calibrated_prob: Platt-scaled probability P(relevant).
            uncertainty_scalar: Mean std from MC Dropout [0, ~0.15].
            gap_score: Top-1 vs this result gap score [0, 1].
            raw_similarity: Raw cosine similarity from FAISS.
            modality_consistency: Cross-modal alignment score [0, 1].
            source_modality: Query modality (for flag detection).
            target_modality: Result modality (for flag detection).
            query_season: Season of query image (for flag detection).
            result_season: Season of result scene (for flag detection).
            query_axis_scores: Dict of axis scores (for boundary detection).

        Returns:
            Fully populated ConfidenceBundle.
        """
        # Normalise uncertainty to [0, 1]
        uncertainty_norm = min(1.0, uncertainty_scalar / CompositeScorer._MAX_UNCERTAINTY)

        # Weighted composite
        score = (
            settings.CONFIDENCE_WEIGHT_CALIB_PROB * calibrated_prob
            + settings.CONFIDENCE_WEIGHT_UNCERTAINTY * (1.0 - uncertainty_norm)
            + settings.CONFIDENCE_WEIGHT_GAP * gap_score
            + settings.CONFIDENCE_WEIGHT_MODALITY * modality_consistency
        )
        score = max(0.0, min(1.0, score))

        # Assign tier
        tier = CompositeScorer._assign_tier(score)

        # Detect flags
        flags = CompositeScorer._detect_flags(
            uncertainty_scalar=uncertainty_scalar,
            source_modality=source_modality,
            target_modality=target_modality,
            modality_consistency=modality_consistency,
            query_season=query_season,
            result_season=result_season,
            query_axis_scores=query_axis_scores,
        )

        # Uncertainty bound string (1-sigma, 95% CI = 1.96σ)
        uncertainty_bound = f"±{uncertainty_scalar * 1.96:.2f}"

        return ConfidenceBundle(
            raw_similarity=round(raw_similarity, 4),
            calibrated_prob=round(calibrated_prob, 4),
            uncertainty=round(uncertainty_scalar, 4),
            similarity_gap=round(gap_score, 4),
            confidence_score=round(score, 4),
            confidence_tier=tier,
            uncertainty_bound=uncertainty_bound,
            flags=flags,
        )

    @staticmethod
    def _assign_tier(score: float) -> ConfidenceTierEnum:
        """Map composite score to confidence tier enum."""
        if score >= 0.80:
            return ConfidenceTierEnum.HIGH
        elif score >= 0.60:
            return ConfidenceTierEnum.MEDIUM
        elif score >= 0.40:
            return ConfidenceTierEnum.LOW
        else:
            return ConfidenceTierEnum.UNCERTAIN

    @staticmethod
    def _detect_flags(
        uncertainty_scalar: float,
        source_modality: str,
        target_modality: str,
        modality_consistency: float,
        query_season: str | None,
        result_season: str | None,
        query_axis_scores: dict | None,
    ) -> list[str]:
        """Detect and return applicable confidence flags."""
        flags = []

        if source_modality != target_modality and modality_consistency < 0.5:
            flags.append("cross_modal_gap")

        if uncertainty_scalar > 0.08:
            flags.append("sparse_region")

        if query_axis_scores:
            values = sorted(query_axis_scores.values(), reverse=True)
            if len(values) >= 2 and (values[0] - values[1]) < 0.10:
                flags.append("near_class_boundary")

        if query_season and result_season and query_season != result_season:
            flags.append("seasonal_mismatch")

        return flags
