"""
GeoSense-X — DINOv2 Encoder Service
=====================================
Loads a pretrained DINOv2 backbone and exposes encode() for
optical and SAR satellite images.

Optical pipeline:
    PIL (RGB) → resize 224×224 → ImageNet normalise → DINOv2 → CLS token → L2 norm

SAR pipeline:
    rasterio (VV/VH bands) → dB convert → pseudo-RGB [VV|VH|diff] →
    ImageNet normalise → DINOv2 → CLS token → L2 norm
    Falls back to PIL grayscale→RGB if rasterio fails.

No projection head or fine-tuning: we use raw DINOv2 CLS features as the
shared embedding space for the Phase 1 MVP.
Cross-modal alignment quality improves after the contrastive training stage.
"""
from __future__ import annotations

import io
import logging
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

Modality = Literal["optical", "sar"]

# ImageNet statistics — used for optical and pseudo-RGB SAR
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Map DINOv2 variant name → expected CLS embedding dimension
_MODEL_DIM: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class EncoderService:
    """
    Wraps a pretrained DINOv2 model for satellite image feature extraction.

    Thread-safe for inference: model.eval() + torch.no_grad() throughout.
    Single instance shared across all requests (stored in app.state.encoder).
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str | None = None) -> None:
        self.model_name = model_name
        self.embedding_dim: int = _MODEL_DIM.get(model_name, 768)

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._model: torch.nn.Module | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """
        Download (first run) and load the DINOv2 backbone.
        Torch Hub caches the weights in ~/.cache/torch/hub/.
        """
        logger.info("Loading %s on %s …", self.model_name, self.device)
        self._model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=True,
            verbose=False,
        )
        self._model.eval()
        self._model = self._model.to(self.device)
        logger.info(
            "DINOv2 loaded  |  dim=%d  |  device=%s",
            self.embedding_dim,
            self.device,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def encode(self, image_bytes: bytes, modality: Modality) -> np.ndarray:
        """
        Encode raw image bytes into an L2-normalised embedding vector.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            modality:    "optical" or "sar".

        Returns:
            np.ndarray of shape [embedding_dim], dtype float32, ‖v‖₂ = 1.

        Raises:
            RuntimeError:  If called before load().
            ValueError:    If modality is unrecognised.
        """
        if self._model is None:
            raise RuntimeError("Call EncoderService.load() before encode().")

        if modality == "optical":
            tensor = self._preprocess_optical(image_bytes)
        elif modality == "sar":
            tensor = self._preprocess_sar(image_bytes)
        else:
            raise ValueError(f"Unsupported modality: {modality!r}. Use 'optical' or 'sar'.")

        return self._forward(tensor)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _preprocess_optical(self, image_bytes: bytes) -> torch.Tensor:
        """RGB image → [1, 3, 224, 224] normalised tensor."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image as optical: {exc}") from exc
        return _TRANSFORM(img).unsqueeze(0).to(self.device)

    def _preprocess_sar(self, image_bytes: bytes) -> torch.Tensor:
        """
        SAR image → pseudo-RGB → [1, 3, 224, 224] normalised tensor.

        Priority order:
          1. rasterio MemoryFile (real GeoTIFF with VV/VH bands)
          2. PIL fallback (grayscale or 3-channel visual)
        """
        img = self._decode_sar_to_pil(image_bytes)
        return _TRANSFORM(img).unsqueeze(0).to(self.device)

    @staticmethod
    def _decode_sar_to_pil(image_bytes: bytes) -> Image.Image:
        """Convert SAR bytes → RGB PIL Image via pseudo-RGB encoding."""
        # ── Attempt rasterio (real GeoTIFF SAR) ─────────────────────────
        try:
            import rasterio
            from rasterio.io import MemoryFile

            with MemoryFile(image_bytes) as memfile, memfile.open() as src:
                data = src.read().astype(np.float32)   # [bands, H, W]

            if data.shape[0] >= 2:
                vv, vh = data[0], data[1]

                # Linear → dB, clip to [-30, 0] dB
                vv_db = np.clip(10.0 * np.log10(np.clip(vv, 1e-10, None)), -30.0, 0.0)
                vh_db = np.clip(10.0 * np.log10(np.clip(vh, 1e-10, None)), -30.0, 0.0)
                diff  = vv_db - vh_db

                def to_uint8(band: np.ndarray) -> np.ndarray:
                    b = band - band.min()
                    return (b / (b.max() + 1e-8) * 255.0).astype(np.uint8)

                rgb = np.stack([to_uint8(vv_db), to_uint8(vh_db), to_uint8(diff)], axis=-1)
                return Image.fromarray(rgb, mode="RGB")

            # Single-band SAR → grayscale → RGB
            b = data[0]
            b = ((b - b.min()) / (b.max() - b.min() + 1e-8) * 255.0).astype(np.uint8)
            return Image.fromarray(np.stack([b, b, b], axis=-1), mode="RGB")

        except Exception:
            pass

        # ── PIL fallback ─────────────────────────────────────────────────
        try:
            pil = Image.open(io.BytesIO(image_bytes))
            if pil.mode == "RGB":
                return pil
            gray = np.array(pil.convert("L"))
            return Image.fromarray(np.stack([gray, gray, gray], axis=-1), mode="RGB")
        except Exception as exc:
            raise ValueError(f"Cannot decode SAR image: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def _forward(self, tensor: torch.Tensor) -> np.ndarray:
        """Run DINOv2 forward pass, extract CLS token, L2 normalise."""
        with torch.no_grad():
            cls_token = self._model(tensor)          # [1, embedding_dim]

        emb = cls_token.squeeze(0).cpu().float().numpy()  # [embedding_dim]
        norm = np.linalg.norm(emb)
        return (emb / (norm + 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
