"""
GeoSense-X — Monte Carlo Dropout Uncertainty Estimator
========================================================
Runs T stochastic forward passes through the encoder with dropout enabled
to estimate embedding uncertainty. Uncertainty is the mean std across dims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


@dataclass
class MCDropoutResult:
    """Result of MC Dropout estimation."""
    z_mean: np.ndarray        # [128] mean embedding across T passes
    z_std: np.ndarray         # [128] std across T passes
    uncertainty_scalar: float # mean(z_std) — the summary uncertainty measure
    n_passes: int


class MCDropoutEstimator:
    """Estimates embedding uncertainty via Monte Carlo Dropout."""

    @staticmethod
    def estimate(
        model: "torch.nn.Module",
        image_tensor: "torch.Tensor",
        modality: str,
        T: int = 5,
    ) -> MCDropoutResult:
        """
        Run T forward passes with dropout enabled to estimate uncertainty.

        Args:
            model: CrossModalModel with Dropout layers in ProjectionHead.
            image_tensor: [1, 3, 224, 224] preprocessed tensor.
            modality: Source modality for encoder selection.
            T: Number of stochastic passes (5 for real-time, 20 for async).

        Returns:
            MCDropoutResult with mean embedding and uncertainty scalar.
        """
        # TODO: implement MC Dropout
        # import torch
        # model.train()  # enables dropout
        # embeddings = []
        # with torch.no_grad():
        #     for _ in range(T):
        #         z = model.encode_single(image_tensor, modality)
        #         embeddings.append(z.cpu().numpy().squeeze())
        # model.eval()  # disable dropout
        # z_stack = np.stack(embeddings)  # [T, 128]
        # z_mean = z_stack.mean(axis=0)   # [128]
        # z_std = z_stack.std(axis=0)     # [128]
        # uncertainty = float(z_std.mean())
        # # Re-normalise mean embedding
        # z_mean = z_mean / (np.linalg.norm(z_mean) + 1e-8)
        # return MCDropoutResult(z_mean=z_mean, z_std=z_std,
        #                        uncertainty_scalar=uncertainty, n_passes=T)
        raise NotImplementedError("MCDropoutEstimator.estimate() not yet implemented")
