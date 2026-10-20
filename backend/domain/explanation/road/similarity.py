"""
GeoSense-X — Road Network Similarity
========================================
Combines edge-based and embedding-axis signals into road similarity score.
"""

from __future__ import annotations

import numpy as np

from domain.explanation.axes.scorer import AxisScorer
from domain.explanation.road.line_detector import LineDetector, RoadFeatureVector


class RoadSimilarity:
    """Computes road network similarity for a (query, result) pair."""

    @staticmethod
    def compute(
        query_features: RoadFeatureVector,
        result_features: RoadFeatureVector,
        query_embedding: np.ndarray,
        result_embedding: np.ndarray,
        edge_weight: float = 0.5,
        axis_weight: float = 0.5,
    ) -> float:
        """
        Compute combined road similarity score.

        Args:
            query_features: Road feature vector for query image.
            result_features: Road feature vector for result image.
            query_embedding: [128] L2-normalised query embedding.
            result_embedding: [128] L2-normalised result embedding.
            edge_weight: Weight for edge-based similarity component.
            axis_weight: Weight for embedding axis similarity component.

        Returns:
            Road similarity float in [0, 1].
        """
        # TODO: implement combined similarity
        # edge_sim = cls._edge_similarity(query_features, result_features)
        # axis_sim = cls._axis_similarity(query_embedding, result_embedding)
        # return edge_weight * edge_sim + axis_weight * axis_sim
        raise NotImplementedError("RoadSimilarity.compute() not yet implemented")

    @staticmethod
    def _edge_similarity(q: RoadFeatureVector, r: RoadFeatureVector) -> float:
        """L2 distance between feature vectors → similarity."""
        q_arr = LineDetector.as_array(q)
        r_arr = LineDetector.as_array(r)
        dist = float(np.linalg.norm(q_arr - r_arr))
        return max(0.0, 1.0 - dist / np.sqrt(4.0))

    @staticmethod
    def _axis_similarity(q_emb: np.ndarray, r_emb: np.ndarray) -> float:
        """Road axis score difference → similarity."""
        q_scores = AxisScorer.score(q_emb)
        r_scores = AxisScorer.score(r_emb)
        return 1.0 - abs(q_scores["road_network"] - r_scores["road_network"])
