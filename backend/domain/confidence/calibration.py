"""
GeoSense-X — Temperature Scaling Calibration
==============================================
Maps raw cosine similarity to calibrated probability P(relevant) using
per-LCLU-class temperature scaling (Platt scaling variant).
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)

# Global fallback parameters when class-specific params are not available
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_THRESHOLD = 0.65


class CalibrationTable:
    """
    In-memory table of temperature scaling parameters per LCLU class.

    Loaded from DB at startup. Provides O(1) lookup for calibrate().
    """

    def __init__(self, params: dict[str, dict]) -> None:
        """
        Initialise with a dict of params.

        Args:
            params: Dict mapping lclu_class → {"temperature": float, "threshold": float}.
        """
        self._params = params

    @classmethod
    async def load(cls, model_version: str) -> "CalibrationTable":
        """
        Load calibration params from DB for the given model version.

        Args:
            model_version: Model version to load params for.

        Returns:
            Populated CalibrationTable instance.
        """
        # TODO: implement DB load
        # from repositories.calibration_repo import CalibrationRepository
        # records = await CalibrationRepository(db).get_all(model_version)
        # params = {r.lclu_class: {"temperature": r.temperature, "threshold": r.threshold}
        #           for r in records}
        # return cls(params)
        logger.warning("CalibrationTable: using default params (DB not yet loaded)")
        return cls({})

    def calibrate(self, similarity: float, lclu_class: str) -> float:
        """
        Convert raw similarity to calibrated probability P(relevant).

        Args:
            similarity: Raw cosine similarity from FAISS [0, 1].
            lclu_class: Predicted LCLU class for this result.

        Returns:
            Calibrated probability in [0, 1].
        """
        params = self._params.get(lclu_class, {})
        temperature = params.get("temperature", _DEFAULT_TEMPERATURE)
        threshold = params.get("threshold", _DEFAULT_THRESHOLD)

        # Platt scaling: sigmoid((sim - threshold) / temperature)
        raw = (similarity - threshold) / (temperature + 1e-8)
        return float(1.0 / (1.0 + np.exp(-raw)))
