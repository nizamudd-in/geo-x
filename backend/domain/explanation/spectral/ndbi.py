"""
GeoSense-X — NDBI Computation
================================
Normalized Difference Built-up Index from multispectral bands.
SAR urban detection via double-bounce backscatter proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NDBIResult:
    """Output of NDBI computation."""
    ndbi_mean: float
    urban_profile: list[float]  # [dense_urban, sparse_urban, bare_soil, vegetated]
    sar_derived: bool
    ndbi_map: np.ndarray | None


class NDBIComputer:
    """Computes NDBI and urban coverage profile."""

    _SWIR_BAND_IDX = 10  # B11 (1610nm)
    _NIR_BAND_IDX = 7    # B8 (842nm)

    # SAR dB thresholds for urban detection (high double-bounce)
    _SAR_URBAN_VV_THRESHOLD = -5.0   # dB
    _SAR_VH_VV_RATIO_THRESHOLD = 0.5

    @classmethod
    def compute(cls, image_array: np.ndarray, modality: str) -> NDBIResult | None:
        """
        Compute NDBI or SAR urban proxy.

        Args:
            image_array: [bands, H, W] image array.
            modality: "optical" | "sar" | "multispectral".

        Returns:
            NDBIResult or None if computation not possible.
        """
        if modality == "multispectral":
            return cls._ndbi_from_ms(image_array)
        elif modality == "sar":
            return cls._urban_from_sar(image_array)
        return None

    @classmethod
    def _ndbi_from_ms(cls, bands: np.ndarray) -> NDBIResult:
        """Compute NDBI = (SWIR - NIR) / (SWIR + NIR)."""
        # TODO: implement NDBI
        # swir = bands[cls._SWIR_BAND_IDX].astype(np.float32)
        # nir = bands[cls._NIR_BAND_IDX].astype(np.float32)
        # ndbi = (swir - nir) / (swir + nir + 1e-8)
        raise NotImplementedError("NDBI not yet implemented")

    @classmethod
    def _urban_from_sar(cls, bands: np.ndarray) -> NDBIResult:
        """SAR urban proxy from high VV backscatter and VH/VV ratio."""
        # TODO: implement SAR urban detection
        raise NotImplementedError("SAR urban detection not yet implemented")

    @staticmethod
    def _compute_profile(ndbi: np.ndarray) -> list[float]:
        """Compute urban coverage profile from NDBI map."""
        total = ndbi.size
        dense = float(np.sum(ndbi > 0.2) / total)
        sparse = float(np.sum((ndbi > 0.0) & (ndbi <= 0.2)) / total)
        bare = float(np.sum((ndbi > -0.2) & (ndbi <= 0.0)) / total)
        vegetated = float(np.sum(ndbi <= -0.2) / total)
        return [dense, sparse, bare, vegetated]
