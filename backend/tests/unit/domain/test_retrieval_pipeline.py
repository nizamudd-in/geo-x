"""
Unit tests for the RetrievalPipeline domain module.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_ranker_deduplicates_by_scene_id() -> None:
    """Ranker should remove duplicate scene_ids keeping highest similarity."""
    # TODO: implement once Ranker is built
    # from domain.retrieval.ranker import Ranker
    # candidates = [
    #     SearchResult(faiss_id=0, scene_id="s1", modality="optical", similarity=0.9),
    #     SearchResult(faiss_id=1, scene_id="s1", modality="optical", similarity=0.8),
    #     SearchResult(faiss_id=2, scene_id="s2", modality="optical", similarity=0.7),
    # ]
    # ranked = Ranker.rerank(np.zeros(128), candidates, top_k=10)
    # scene_ids = [r.scene_id for r in ranked]
    # assert scene_ids.count("s1") == 1
    # assert ranked[0].scene_id == "s1"  # kept highest sim
    pytest.skip("Ranker not yet implemented")


def test_ranker_respects_top_k() -> None:
    """Ranker should return at most top_k results."""
    pytest.skip("Ranker not yet implemented")


def test_modality_router_same_modal_uses_per_modality_index() -> None:
    """ModalityRouter should use per-modality index for same-modal search."""
    pytest.skip("ModalityRouter not yet implemented")


def test_modality_router_cross_modal_uses_shared_index() -> None:
    """ModalityRouter should use shared index for cross-modal search."""
    pytest.skip("ModalityRouter not yet implemented")
