"""
GeoSense-X — DINOv2 Attention Extractor
==========================================
Registers a forward hook on DINOv2 Layer 23 to capture
multi-head self-attention weights during inference.

Outputs [CLS] → patch attention for all 16 heads.
Used by both the heatmap renderer and rollout calculator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.logging import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class AttentionExtractor:
    """
    Extracts [CLS] → patch attention from DINOv2 ViT-L/14.

    Registers a hook on the final transformer layer and captures
    the attention weight tensor during a forward pass.
    Thread-safe: hook is registered and removed per call.
    """

    @staticmethod
    def extract(
        model: "torch.nn.Module",
        image_tensor: "torch.Tensor",
        layer_idx: int = 23,
        head_aggregation: str = "mean",
    ) -> np.ndarray:
        """
        Run a forward pass and extract [CLS] → patch attention.

        Args:
            model: Loaded DINOv2 model (CrossModalModel or backbone directly).
            image_tensor: [1, 3, 224, 224] preprocessed input tensor.
            layer_idx: Transformer layer to extract from (default: 23, final layer).
            head_aggregation: How to aggregate 16 attention heads:
                "mean" — average across heads (stable, used in fast mode)
                "max"  — max across heads (highlights salient patches)
                "none" — return all heads [16, 324]

        Returns:
            np.ndarray [324] (or [16, 324] if head_aggregation="none")
            representing [CLS] token attention to each of 18×18=324 patch tokens.

        Raises:
            ValueError: If image_tensor has wrong shape.
        """
        # TODO: implement attention extraction
        # captured = {}
        #
        # def hook_fn(module, input, output):
        #     # output is (attn_output, attn_weights)
        #     captured["attn"] = output[1].detach()
        #
        # target_layer = model.backbone.blocks[layer_idx].attn
        # handle = target_layer.register_forward_hook(hook_fn)
        #
        # try:
        #     with torch.no_grad():
        #         _ = model.backbone(image_tensor)
        # finally:
        #     handle.remove()
        #
        # attn = captured["attn"]  # [1, 16, 325, 325]
        # cls_to_patch = attn[0, :, 0, 1:]  # [16, 324] — CLS row, skip CLS column
        #
        # if head_aggregation == "mean":
        #     return cls_to_patch.mean(0).cpu().numpy()  # [324]
        # elif head_aggregation == "max":
        #     return cls_to_patch.max(0).values.cpu().numpy()
        # else:
        #     return cls_to_patch.cpu().numpy()  # [16, 324]

        raise NotImplementedError("AttentionExtractor.extract() not yet implemented")
