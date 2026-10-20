"""
GeoSense-X — Image Preprocessor
==================================
Converts raw image bytes into model-ready tensors.
Three modality-specific pipelines, unified behind a single preprocess() call.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from config import settings
from core.exceptions import PreprocessingError
from core.logging import get_logger

logger = get_logger(__name__)

# Expected output: [1, 3, 224, 224] float32 tensor for all modalities
_TARGET_SIZE = (224, 224)

# ImageNet normalisation constants (optical)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# SAR normalisation constants (dB scale, empirical from SEN12MS)
_SAR_MEAN = (-12.0, -19.5, 7.5)  # VV, VH, VV-VH
_SAR_STD = (4.5, 4.2, 3.8)

# MS normalisation (per-band, fit on SEN12MS training set)
# TODO: load from pca_checkpoint once available
_MS_MEAN = [0.0] * 3  # placeholder — computed by PCA projection output
_MS_STD = [1.0] * 3


class Preprocessor:
    """
    Stateless image preprocessor with one class method per modality.

    PCA projector is loaded once as a class-level attribute and reused
    across all requests.
    """

    _pca_matrix: np.ndarray | None = None
    _pca_mean: np.ndarray | None = None

    @classmethod
    def _load_pca(cls) -> None:
        """Load PCA projection matrix from checkpoint (lazy, called once)."""
        if cls._pca_matrix is not None:
            return
        pca_path = settings.PCA_CHECKPOINT
        if not Path(pca_path).exists():
            logger.warning("PCA checkpoint not found", path=str(pca_path))
            return
        # TODO: implement PCA loading
        # data = np.load(pca_path, allow_pickle=True).item()
        # cls._pca_matrix = data["components"]  # [3 × 13]
        # cls._pca_mean = data["mean"]           # [13]

    @classmethod
    def preprocess(cls, image_bytes: bytes, modality: str) -> "torch.Tensor":
        """
        Convert raw image bytes to a normalised [1, 3, 224, 224] tensor.

        Routes to modality-specific pipeline.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            modality: One of "optical", "sar", "multispectral".

        Returns:
            torch.Tensor [1, 3, 224, 224] float32, ready for DINOv2 input.

        Raises:
            PreprocessingError: If image cannot be decoded or is wrong format.
            ValueError: If modality is unknown.
        """
        if modality == "optical":
            return cls._preprocess_optical(image_bytes)
        elif modality == "sar":
            return cls._preprocess_sar(image_bytes)
        elif modality == "multispectral":
            return cls._preprocess_multispectral(image_bytes)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    @classmethod
    def _preprocess_optical(cls, image_bytes: bytes) -> "torch.Tensor":
        """
        Optical preprocessing pipeline:
          PIL decode → RGB → resize 224×224 → ToTensor → ImageNet normalise.
        """
        # TODO: implement optical preprocessing
        # from PIL import Image
        # import torchvision.transforms.functional as TF
        # img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # img = img.resize(_TARGET_SIZE, Image.LANCZOS)
        # tensor = TF.to_tensor(img)  # [3, 224, 224]
        # tensor = TF.normalize(tensor, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        # return tensor.unsqueeze(0)  # [1, 3, 224, 224]
        raise NotImplementedError("Optical preprocessing not yet implemented")

    @classmethod
    def _preprocess_sar(cls, image_bytes: bytes) -> "torch.Tensor":
        """
        SAR preprocessing pipeline:
          rasterio decode → read 2 bands → dB convert → Lee filter →
          clip [-30, 0] → pseudo-RGB [VV, VH, VV-VH] → normalise.
        """
        # TODO: implement SAR preprocessing
        # import rasterio, torch
        # with rasterio.open(io.BytesIO(image_bytes)) as src:
        #     bands = src.read()  # [2, H, W]
        # vv = 10 * np.log10(np.clip(bands[0], 1e-10, None))
        # vh = 10 * np.log10(np.clip(bands[1], 1e-10, None))
        # Apply Lee speckle filter (scipy convolution)
        # vv_vh_diff = vv - vh
        # pseudo_rgb = np.stack([vv, vh, vv_vh_diff])
        # Normalise and convert to tensor
        raise NotImplementedError("SAR preprocessing not yet implemented")

    @classmethod
    def _preprocess_multispectral(cls, image_bytes: bytes) -> "torch.Tensor":
        """
        Multispectral preprocessing pipeline:
          rasterio decode → read 13 bands → PCA project (13→3) → normalise.
        """
        cls._load_pca()
        # TODO: implement MS preprocessing
        # import rasterio, torch
        # with rasterio.open(io.BytesIO(image_bytes)) as src:
        #     bands = src.read()  # [13, H, W]
        # if cls._pca_matrix is not None:
        #     flat = bands.reshape(13, -1).T  # [H*W, 13]
        #     projected = (flat - cls._pca_mean) @ cls._pca_matrix.T  # [H*W, 3]
        #     projected = projected.T.reshape(3, H, W)
        # Normalise and resize
        raise NotImplementedError("Multispectral preprocessing not yet implemented")
