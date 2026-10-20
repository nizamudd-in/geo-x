"""
GeoSense-X — NDVI Computation
================================
Normalized Difference Vegetation Index from multispectral or optical bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NDVIResult:
    """Output of NDVI computation."""
    ndvi_mean: float
    ndvi_std: float
    coverage_profile: list[float]  # [dense_forest, moderate_veg, sparse_veg, non_veg]
    approximated: bool             # True if computed from non-NIR bands
    ndvi_map: np.ndarray | None    # [H, W] per-pixel NDVI (None if not computed)


class NDVIComputer:
    """Computes NDVI and vegetation coverage profile from image arrays."""

    # SEN12MS Sentinel-2 band indices (0-based)
    _NIR_BAND_IDX = 7   # B8 (842nm)
    _RED_BAND_IDX = 3   # B4 (665nm)
    _GREEN_BAND_IDX = 2  # B3 (560nm) — used as NIR proxy for optical

    @classmethod
    def compute(cls, image_array: np.ndarray, modality: str) -> NDVIResult | None:
        """
        Compute NDVI from a raw image array.

        Args:
            image_array: For MS: [13, H, W]. For optical: [3, H, W] RGB.
                         For SAR: not applicable, returns None.
            modality: "optical" | "sar" | "multispectral".

        Returns:
            NDVIResult or None if modality is SAR.
        """
        if modality == "sar":
            return None

        if modality == "multispectral":
            return cls._ndvi_from_multispectral(image_array)
        else:
            return cls._ndvi_from_optical(image_array)

    @classmethod
    def _ndvi_from_multispectral(cls, bands: np.ndarray) -> NDVIResult:
        """Compute NDVI from 13-band Sentinel-2 array."""
        # TODO: implement NDVI from MS bands
        # nir = bands[cls._NIR_BAND_IDX].astype(np.float32)
        # red = bands[cls._RED_BAND_IDX].astype(np.float32)
        # ndvi = (nir - red) / (nir + red + 1e-8)
        # ndvi = np.clip(ndvi, -1.0, 1.0)
        # profile = cls._compute_profile(ndvi)
        # return NDVIResult(
        #     ndvi_mean=float(ndvi.mean()),
        #     ndvi_std=float(ndvi.std()),
        #     coverage_profile=profile,
        #     approximated=False,
        #     ndvi_map=ndvi,
        # )
        raise NotImplementedError("NDVI from multispectral not yet implemented")

    @classmethod
    def _ndvi_from_optical(cls, bands: np.ndarray) -> NDVIResult:
        """Approximate NDVI from RGB (uses Green as NIR proxy — weaker)."""
        # TODO: implement NDVI approximation from RGB
        raise NotImplementedError("NDVI from optical not yet implemented")

    @staticmethod
    def _compute_profile(ndvi: np.ndarray) -> list[float]:
        """Compute coverage profile from NDVI map."""
        total = ndvi.size
        dense = float(np.sum(ndvi > 0.6) / total)
        moderate = float(np.sum((ndvi > 0.3) & (ndvi <= 0.6)) / total)
        sparse = float(np.sum((ndvi > 0.1) & (ndvi <= 0.3)) / total)
        non_veg = float(np.sum(ndvi <= 0.1) / total)
        return [dense, moderate, sparse, non_veg]
