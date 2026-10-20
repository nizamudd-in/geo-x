"""
Unit tests for the ConfidencePipeline and calibration components.
"""

from __future__ import annotations

import pytest


def test_calibration_returns_high_prob_for_high_similarity() -> None:
    """CalibrationTable.calibrate() should return > 0.7 for similarity > threshold."""
    from domain.confidence.calibration import CalibrationTable

    table = CalibrationTable(
        params={"urban_fabric": {"temperature": 0.5, "threshold": 0.65}}
    )
    prob = table.calibrate(0.90, "urban_fabric")
    assert prob > 0.70


def test_calibration_uses_default_for_unknown_class() -> None:
    """CalibrationTable should use global defaults for unregistered LCLU classes."""
    from domain.confidence.calibration import CalibrationTable

    table = CalibrationTable(params={})
    prob = table.calibrate(0.80, "nonexistent_class")
    assert 0.0 <= prob <= 1.0


def test_composite_scorer_flags_sparse_region() -> None:
    """CompositeScorer should set sparse_region flag when uncertainty > 0.08."""
    from domain.confidence.composite import CompositeScorer

    bundle = CompositeScorer.compute(
        calibrated_prob=0.70,
        uncertainty_scalar=0.12,  # above 0.08 threshold
        gap_score=0.70,
        raw_similarity=0.75,
    )
    assert "sparse_region" in bundle.flags


def test_composite_scorer_no_flags_for_ideal_input() -> None:
    """CompositeScorer should return no flags for a clean same-modal result."""
    from domain.confidence.composite import CompositeScorer

    bundle = CompositeScorer.compute(
        calibrated_prob=0.90,
        uncertainty_scalar=0.02,
        gap_score=0.85,
        raw_similarity=0.90,
        source_modality="optical",
        target_modality="optical",
        query_season="summer",
        result_season="summer",
    )
    assert bundle.flags == []
