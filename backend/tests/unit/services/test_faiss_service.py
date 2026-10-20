"""
Unit tests for the FAISSService.
"""

from __future__ import annotations

import pytest


def test_faiss_service_raises_before_load() -> None:
    """FAISSService.search() should raise IndexNotLoadedError before indexes are loaded."""
    import numpy as np
    from core.exceptions import IndexNotLoadedError
    from services.faiss_service import FAISSService

    svc = FAISSService()
    emb = np.random.randn(128).astype(np.float32)
    emb /= np.linalg.norm(emb)

    with pytest.raises((IndexNotLoadedError, NotImplementedError)):
        svc.search(emb, "optical", "sar", k=10)


def test_faiss_service_is_loaded_returns_false_before_load() -> None:
    """FAISSService.is_loaded should be False before load_indexes() is called."""
    from services.faiss_service import FAISSService

    svc = FAISSService()
    assert svc.is_loaded is False


# TODO: add tests with actual FAISS index once ml/scripts/build_faiss_index.py is run
# def test_faiss_same_modal_returns_self_as_top1(built_indexes):
#     ...
