"""
GeoSense-X — Hough Line Detector
====================================
Detects linear road-like features from edge maps using probabilistic Hough transform.
Extracts: line count, mean length, orientation entropy (grid vs. highway pattern).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoadFeatureVector:
    """Four road network features extracted from an image."""
    edge_density: float          # fraction of edge pixels
    line_count_norm: float       # line count normalised by 100
    mean_line_length_norm: float # mean line length normalised by 256
    orientation_entropy_norm: float  # orientation entropy normalised by log2(8)


class LineDetector:
    """Detects linear features and extracts road network statistics."""

    @staticmethod
    def detect(
        edge_map: np.ndarray,
        edge_density: float,
        min_line_length: int = 15,
        max_line_gap: int = 5,
    ) -> RoadFeatureVector:
        """
        Detect line segments in an edge map and extract road features.

        Args:
            edge_map: [H, W] binary edge map from EdgeDetector.
            edge_density: Pre-computed edge density scalar.
            min_line_length: Minimum line segment length in pixels.
            max_line_gap: Maximum gap in pixels to bridge when detecting lines.

        Returns:
            RoadFeatureVector with normalised road features.
        """
        # TODO: implement Hough line transform
        # import cv2
        # edge_uint8 = (edge_map * 255).astype(np.uint8)
        # lines = cv2.HoughLinesP(edge_uint8, rho=1, theta=np.pi/180,
        #                          threshold=30, minLineLength=min_line_length,
        #                          maxLineGap=max_line_gap)
        # if lines is None:
        #     return RoadFeatureVector(edge_density, 0.0, 0.0, 0.0)
        # Compute features
        # lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in lines[:,0]]
        # angles = [np.degrees(np.arctan2(y2-y1, x2-x1)) % 180 for x1,y1,x2,y2 in lines[:,0]]
        # Quantise angles to 8 bins and compute entropy
        # hist, _ = np.histogram(angles, bins=8, range=(0, 180))
        # hist = hist / (hist.sum() + 1e-8)
        # entropy = -np.sum(hist * np.log2(hist + 1e-8))
        # return RoadFeatureVector(
        #     edge_density=edge_density,
        #     line_count_norm=min(len(lines) / 100, 1.0),
        #     mean_line_length_norm=min(np.mean(lengths) / 256, 1.0),
        #     orientation_entropy_norm=min(entropy / np.log2(8), 1.0),
        # )
        raise NotImplementedError("LineDetector.detect() not yet implemented")

    @staticmethod
    def as_array(features: RoadFeatureVector) -> np.ndarray:
        """Convert RoadFeatureVector to [4] numpy array."""
        return np.array([
            features.edge_density,
            features.line_count_norm,
            features.mean_line_length_norm,
            features.orientation_entropy_norm,
        ], dtype=np.float32)
