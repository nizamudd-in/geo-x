"""
GeoSense-X — Semantic Axis Scorer
====================================
Computes axis scores for any embedding using learned linear probe direction vectors.

Axis vectors are loaded from DB once at startup and cached as module-level
numpy arrays. Thread-safe (read-only arrays, no mutation after load).
"""

from __future__ import annotations

import json

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)

# Module-level axis vector cache (loaded once at startup)
_AXIS_VECTORS: dict[str, np.ndarray] = {}
_AXIS_BIASES: dict[str, float] = {}
_CURRENT_MODEL_VERSION: str = ""

AXIS_NAMES = ("vegetation", "waterbody", "urban", "road_network")


class AxisScorer:
    """
    Computes semantic axis scores for any 128-d L2-normalised embedding.

    Scores represent the strength of each semantic property in the scene:
      - vegetation:   0=no vegetation, 1=dense forest cover
      - waterbody:    0=no water, 1=open water dominant
      - urban:        0=rural, 1=dense urban fabric
      - road_network: 0=no roads, 1=high road density
    """

    @classmethod
    async def load_axis_vectors(cls, model_version: str) -> None:
        """
        Load axis direction vectors from DB into module-level cache.

        Called once at startup after DB connection is established.
        Also called after model hot-reload to update for new version.

        Args:
            model_version: Current model version to load vectors for.
        """
        global _AXIS_VECTORS, _AXIS_BIASES, _CURRENT_MODEL_VERSION

        # TODO: implement DB load via axis_vector ORM model
        # from repositories.axis_vector_repo import AxisVectorRepository
        # vectors = await AxisVectorRepository(db).get_all(model_version)
        # for v in vectors:
        #     _AXIS_VECTORS[v.axis_name] = np.array(json.loads(v.axis_vector_json))
        #     _AXIS_BIASES[v.axis_name] = v.bias
        # _CURRENT_MODEL_VERSION = model_version
        # logger.info("Axis vectors loaded", model_version=model_version, count=len(vectors))

        logger.warning("Axis vectors not loaded — DB load not yet implemented")

    @staticmethod
    def score(embedding: np.ndarray) -> dict[str, float]:
        """
        Compute all four axis scores for a single embedding.

        Args:
            embedding: [128] float32 L2-normalised embedding vector.

        Returns:
            Dict mapping axis name → score in [0, 1].
            Returns 0.5 (neutral) for any axis not yet loaded.
        """
        scores: dict[str, float] = {}
        for axis in AXIS_NAMES:
            if axis not in _AXIS_VECTORS:
                scores[axis] = 0.5  # neutral fallback
                continue
            # TODO: implement dot product + sigmoid
            # raw = float(np.dot(embedding, _AXIS_VECTORS[axis])) + _AXIS_BIASES[axis]
            # scores[axis] = float(1.0 / (1.0 + np.exp(-raw)))
            scores[axis] = 0.5  # placeholder
        return scores

    @staticmethod
    def score_batch(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute axis scores for a batch of embeddings.

        Args:
            embeddings: [N, 128] float32 batch of embeddings.

        Returns:
            np.ndarray [N, 4] with columns [vegetation, waterbody, urban, road].
        """
        # TODO: implement batch scoring
        # result = np.zeros((len(embeddings), 4), dtype=np.float32)
        # for i, axis in enumerate(AXIS_NAMES):
        #     if axis in _AXIS_VECTORS:
        #         raw = embeddings @ _AXIS_VECTORS[axis] + _AXIS_BIASES[axis]
        #         result[:, i] = 1.0 / (1.0 + np.exp(-raw))
        #     else:
        #         result[:, i] = 0.5
        # return result
        raise NotImplementedError("score_batch() not yet implemented")

    @staticmethod
    def get_axis_vector(axis_name: str) -> np.ndarray | None:
        """
        Return the raw axis direction vector for use in query refinement.

        Args:
            axis_name: One of the AXIS_NAMES.

        Returns:
            [128] unit vector or None if not loaded.
        """
        return _AXIS_VECTORS.get(axis_name)
