"""
GeoSense-X — Attention Rollout
=================================
Implements the attention rollout algorithm (Abnar & Zuidema, 2020).
Propagates attention through all 24 transformer layers for higher-fidelity
attribution than single-layer extraction.

Used in "full" explanation mode only (adds ~100ms per image).
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


class AttentionRollout:
    """
    Computes attention rollout across all transformer layers.

    Algorithm: A_rollout = Π_{l=0}^{L} (I + 0.5 × A_l)
    where A_l is the mean-head attention matrix for layer l.
    """

    @staticmethod
    def compute(
        all_layer_attentions: list[np.ndarray],
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Compute attention rollout from per-layer attention matrices.

        Args:
            all_layer_attentions: List of 24 attention matrices,
                each [n_heads, seq_len, seq_len] (includes CLS token).
            discard_ratio: Fraction of lowest attentions to zero-out
                per layer (reduces noise propagation).

        Returns:
            np.ndarray [324] float32 — [CLS] attention to all patch tokens
            after rolling attention through all layers.
        """
        # TODO: implement rollout algorithm
        # result = np.eye(all_layer_attentions[0].shape[-1])
        # for attn in all_layer_attentions:
        #     # Average heads
        #     attn_avg = attn.mean(axis=0)  # [seq_len, seq_len]
        #     # Add residual: I + attn
        #     attn_with_residual = attn_avg + np.eye(attn_avg.shape[0])
        #     # Apply discard (zero low-attention tokens)
        #     flat = attn_with_residual.flatten()
        #     threshold = np.percentile(flat, discard_ratio * 100)
        #     attn_with_residual[attn_with_residual < threshold] = 0
        #     # Normalise rows
        #     attn_with_residual /= (attn_with_residual.sum(axis=-1, keepdims=True) + 1e-8)
        #     # Matrix product (rollout)
        #     result = attn_with_residual @ result
        # # Return [CLS] → patch attention (first row, skip CLS column)
        # return result[0, 1:]  # [324]
        raise NotImplementedError("AttentionRollout.compute() not yet implemented")
