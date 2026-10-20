"""
GeoSense-X — NDWI Computation
================================
Normalized Difference Water Index from multispectral bands.
SAR water detection via backscatter thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NDWIResult:
    """Output of NDWI computation."""
    ndwi_mean: float
    water_profile: list[float]  # [open_water, wetland, moist_soil, dry_land]
    sar_derived: bool
    ndwi_map: np.ndarray | None


class NDWIComputer:
    """Computes NDWI and water coverage profile."""

    _GREEN_BAND_IDX = 2  # B3 (560nm)
    _NIR_BAND_IDX = 7    # B8 (842nm)

    # SAR dB thresholds for water detection
    _SAR_WATER_VV_THRESHOLD = -15.0   # dB
    _SAR_WATER_VH_THRESHOLD = -22.0   # dB

    @classmethod
    def compute(cls, image_array: np.ndarray, modality: str) -> NDWIResult | None:
        """
        Compute NDWI or SAR water proxy.

        Args:
            image_array: [bands, H, W] image array.
            modality: "optical" | "sar" | "multispectral".

        Returns:
            NDWIResult or None if computation not possible.
        """
        if modality == "multispectral":
            return cls._ndwi_from_ms(image_array)
        elif modality == "sar":
            return cls._water_from_sar(image_array)
        # optical: no NIR band → cannot compute reliable NDWI
        return None

    @classmethod
    def _ndwi_from_ms(cls, bands: np.ndarray) -> NDWIResult:
        """Compute NDWI = (Green - NIR) / (Green + NIR)."""
        # TODO: implement NDWI from MS bands
        # green = bands[cls._GREEN_BAND_IDX].astype(np.float32)
        # nir = bands[cls._NIR_BAND_IDX].astype(np.float32)
        # ndwi = (green - nir) / (green + nir + 1e-8)
        # ndwi = np.clip(ndwi, -1.0, 1.0)
        # profile = cls._compute_profile(ndwi)
        # return NDWIResult(ndwi_mean=float(ndwi.mean()), water_profile=profile,
        #                   sar_derived=False, ndwi_map=ndwi)
        raise NotImplementedError("NDWI not yet implemented")

    @classmethod
    def _water_from_sar(cls, bands: np.ndarray) -> NDWIResult:
        """Detect water from SAR low backscatter signature."""
        # TODO: SAR water detection
        # vv = bands[0]  # dB values
        # vh = bands[1]
        # water_mask = (vv < cls._SAR_WATER_VV_THRESHOLD) & (vh < cls._SAR_WATER_VH_THRESHOLD)
        # water_pct = float(water_mask.mean())
        # profile = [water_pct, 0.0, 0.0, 1.0 - water_pct]
        raise NotImplementedError("SAR water detection not yet implemented")

    @staticmethod
    def _compute_profile(ndwi: np.ndarray) -> list[float]:
        """Compute water coverage profile from NDWI map."""
        total = ndwi.size
        open_water = float(np.sum(ndwi > 0.3) / total)
        wetland = float(np.sum((ndwi > 0.0) & (ndwi <= 0.3)) / total)
        moist = float(np.sum((ndwi > -0.3) & (ndwi <= 0.0)) / total)
        dry = float(np.sum(ndwi <= -0.3) / total)
        return [open_water, wetland, moist, dry]
