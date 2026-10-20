"""
GeoSense-X — Attention Alignment Score
=========================================
Measures structural similarity between query and result attention heatmaps.
Used to quantify how spatially aligned the model's focus is for a pair.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


class AttentionAligner:
    """Computes SSIM-based attention alignment between query and result."""

    @staticmethod
    def score(
        query_attention: np.ndarray,
        result_attention: np.ndarray,
    ) -> tuple[float, str]:
        """
        Compute SSIM between two attention maps.

        Both inputs can be either:
          - [324] flat vectors (from extractor output)
          - [18, 18] compressed grids (from DB storage)

        Args:
            query_attention: Query image attention map.
            result_attention: Result image attention map.

        Returns:
            Tuple of (ssim_score: float, interpretation: str).
            ssim_score is in [0, 1]; 1.0 = identical spatial pattern.
        """
        # TODO: implement SSIM
        # from skimage.metrics import structural_similarity as ssim
        # q = query_attention.reshape(18, 18).astype(np.float32)
        # r = result_attention.reshape(18, 18).astype(np.float32)
        # Normalise to [0, 1]
        # q = (q - q.min()) / (q.max() - q.min() + 1e-8)
        # r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        # score_val = float(ssim(q, r, data_range=1.0))
        # score_val = max(0.0, score_val)  # SSIM can be negative
        # interpretation = _interpret(score_val)
        # return score_val, interpretation
        raise NotImplementedError("AttentionAligner.score() not yet implemented")


def _interpret(ssim_score: float) -> str:
    """Convert SSIM score to a human-readable interpretation string."""
    if ssim_score > 0.70:
        return "Strong spatial alignment — both images focus on similar regions"
    elif ssim_score > 0.40:
        return "Partial alignment — some shared structural focus"
    else:
        return "Low alignment — retrieved by global semantics, not spatial pattern"
