"""
GeoSense-X — Heatmap Renderer
================================
Converts raw attention vectors into visual heatmap artifacts:
  - PNG overlay images for API response
  - float32 arrays for DB storage (18×18 grid)
  - Diff overlay (red=query-only, blue=result-only, green=shared)
"""

from __future__ import annotations

import io

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_OPACITY = 0.65
_DEFAULT_THRESHOLD_PERCENTILE = 40


class HeatmapRenderer:
    """
    Renders a single attention vector as visual heatmap artifacts.

    Instantiated with the raw [324] attention vector.
    Call to_png_bytes(), to_array(), or to_top3_regions().
    """

    def __init__(
        self,
        attention_vector: np.ndarray,
        original_image: np.ndarray | None = None,
    ) -> None:
        """
        Initialise the renderer.

        Args:
            attention_vector: [324] float32 CLS → patch attention.
            original_image: Optional [H, W, 3] uint8 RGB image for overlay.
        """
        self._attention = attention_vector
        self._original = original_image
        self._heatmap_256: np.ndarray | None = None

    def _compute_heatmap_256(self) -> np.ndarray:
        """Upsample 18×18 attention grid to 256×256."""
        if self._heatmap_256 is not None:
            return self._heatmap_256

        # TODO: implement upsampling
        # grid = self._attention.reshape(18, 18)
        # from PIL import Image
        # pil_grid = Image.fromarray((grid * 255).astype(np.uint8))
        # upsampled = pil_grid.resize((256, 256), Image.BILINEAR)
        # heatmap = np.array(upsampled, dtype=np.float32) / 255.0
        # Normalize to [0, 1]
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # self._heatmap_256 = heatmap
        # return heatmap
        raise NotImplementedError("_compute_heatmap_256() not yet implemented")

    def to_png_bytes(
        self,
        colormap: str = "viridis",
        opacity: float = _DEFAULT_OPACITY,
        threshold_percentile: float = _DEFAULT_THRESHOLD_PERCENTILE,
    ) -> bytes:
        """
        Render attention heatmap as PNG overlay bytes.

        Args:
            colormap: Matplotlib colormap name.
            opacity: Overlay opacity [0, 1].
            threshold_percentile: Zero-out pixels below this percentile.

        Returns:
            PNG image as bytes.
        """
        # TODO: implement PNG rendering
        # import matplotlib.cm as cm
        # heatmap = self._compute_heatmap_256()
        # threshold = np.percentile(heatmap, threshold_percentile)
        # heatmap[heatmap < threshold] = 0
        # coloured = cm.get_cmap(colormap)(heatmap)  # [256, 256, 4] RGBA
        # if self._original is not None:
        #     overlay = opacity * coloured[..., :3] + (1 - opacity) * (self._original / 255.0)
        # else:
        #     overlay = coloured[..., :3]
        # img = Image.fromarray((overlay * 255).astype(np.uint8))
        # buf = io.BytesIO()
        # img.save(buf, format="PNG")
        # return buf.getvalue()
        raise NotImplementedError("to_png_bytes() not yet implemented")

    def to_array(self) -> np.ndarray:
        """
        Return the 18×18 compressed attention grid for DB storage.

        Returns:
            np.ndarray [18, 18] float32, min-max normalised.
        """
        # TODO: implement
        raise NotImplementedError

    def to_top3_regions(self, lclu_mask: np.ndarray | None = None) -> list[dict]:
        """
        Identify top-3 attended land cover regions.

        Args:
            lclu_mask: [256, 256] int32 LCLU class per pixel. If None,
                       returns regions based on spatial quadrants only.

        Returns:
            List of up to 3 dicts: {class_name, attention_density, coverage_pct}.
        """
        # TODO: implement region importance extraction
        raise NotImplementedError

    @staticmethod
    def diff_overlay(
        query_heatmap: np.ndarray,
        result_heatmap: np.ndarray,
    ) -> bytes:
        """
        Generate a three-channel diff overlay PNG.

        Channels:
          Red   → query-only attention (high in query, low in result)
          Blue  → result-only attention
          Green → shared attention (high in both)

        Args:
            query_heatmap: [256, 256] float32 query attention.
            result_heatmap: [256, 256] float32 result attention.

        Returns:
            PNG image bytes of the diff overlay.
        """
        # TODO: implement diff rendering
        raise NotImplementedError("diff_overlay() not yet implemented")
