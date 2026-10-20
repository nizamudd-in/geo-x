"""
Unit tests for the ExplanationPipeline and its sub-components.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_axis_comparator_strong_match() -> None:
    """AxisComparator should return 'strong match' when scores are close."""
    from domain.explanation.axes.comparator import AxisComparator

    # Both embeddings are the same — all scores should match perfectly
    emb = np.random.randn(128).astype(np.float32)
    emb /= np.linalg.norm(emb)

    bundle = AxisComparator.compare(emb, emb)
    assert bundle.vegetation.similarity == pytest.approx(1.0, abs=0.01)
    assert bundle.urban.similarity == pytest.approx(1.0, abs=0.01)


def test_narrative_categorizer_assigns_uncertain_on_low_confidence() -> None:
    """NarrativeCategorizer should return UNCERTAIN when confidence < 0.50."""
    from domain.explanation.narrative.categorizer import NarrativeCategorizer
    from domain.explanation.axes.comparator import AxisComparator
    from models.schemas.explanation import NarrativeCategoryEnum

    emb = np.zeros(128, dtype=np.float32)
    bundle = AxisComparator.compare(emb, emb)

    category = NarrativeCategorizer.categorize(
        global_similarity=0.90,
        axis_bundle=bundle,
        confidence_score=0.30,  # below threshold
    )
    assert category == NarrativeCategoryEnum.UNCERTAIN


def test_narrative_generator_produces_non_empty_string() -> None:
    """NarrativeGenerator should return a non-empty string for all categories."""
    from domain.explanation.narrative.generator import NarrativeGenerator
    from domain.explanation.axes.comparator import AxisComparator
    from models.schemas.explanation import NarrativeCategoryEnum, RegionImportance

    emb = np.random.randn(128).astype(np.float32)
    emb /= np.linalg.norm(emb)

    bundle = AxisComparator.compare(emb, emb)
    regions = [RegionImportance(class_name="urban_fabric", attention_density=0.8, coverage_pct=60.0)]

    for category in NarrativeCategoryEnum:
        narrative = NarrativeGenerator.generate(
            global_similarity=0.85,
            axis_bundle=bundle,
            confidence_score=0.75,
            region_importance=regions,
            category=category,
        )
        assert isinstance(narrative, str)
        assert len(narrative) > 10


def test_gap_scorer_rank1_has_no_gap() -> None:
    """GapScorer: rank-1 result should have gap_score = 1.0 (no gap to itself)."""
    from domain.confidence.gap_scorer import GapScorer

    similarities = [0.95, 0.80, 0.70, 0.65, 0.60]
    scores = GapScorer.compute_all(similarities)
    assert scores[0] == pytest.approx(1.0, abs=0.001)


def test_gap_scorer_lower_ranks_have_smaller_scores() -> None:
    """GapScorer: later ranks should have lower gap scores."""
    from domain.confidence.gap_scorer import GapScorer

    similarities = [0.95, 0.80, 0.60]
    scores = GapScorer.compute_all(similarities)
    assert scores[0] >= scores[1] >= scores[2]


def test_composite_scorer_high_confidence() -> None:
    """CompositeScorer should return HIGH tier for ideal inputs."""
    from domain.confidence.composite import CompositeScorer
    from models.schemas.common import ConfidenceTierEnum

    bundle = CompositeScorer.compute(
        calibrated_prob=0.95,
        uncertainty_scalar=0.01,
        gap_score=0.90,
        raw_similarity=0.92,
        modality_consistency=1.0,
    )
    assert bundle.confidence_tier == ConfidenceTierEnum.HIGH
    assert bundle.confidence_score > 0.80


def test_composite_scorer_seasonal_mismatch_flag() -> None:
    """CompositeScorer should set seasonal_mismatch flag when seasons differ."""
    from domain.confidence.composite import CompositeScorer

    bundle = CompositeScorer.compute(
        calibrated_prob=0.80,
        uncertainty_scalar=0.03,
        gap_score=0.75,
        raw_similarity=0.80,
        query_season="summer",
        result_season="winter",
    )
    assert "seasonal_mismatch" in bundle.flags
