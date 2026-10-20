"""
GeoSense-X — DINOv2 Embedding Generation Pipeline
====================================================
Processes SEN12MS scenes through the DINOv2 encoder and saves L2-normalised
embeddings in the format expected by the FAISS service.

Output (written to --output-dir, default: ../backend/embeddings)
────────────────────────────────────────────────────────────────
  optical.npy      [N, D] float32  L2-normalised DINOv2 CLS tokens (optical)
  sar.npy          [N, D] float32  L2-normalised DINOv2 CLS tokens (SAR)
  metadata.json    list of N dicts (scene_id, lat, lon, season, lclu_class)

Row i in optical.npy and sar.npy corresponds to metadata[i].
This alignment is required by services/faiss_service.py → build_indexes().

Pipeline per scene
──────────────────
  Optical:
    Sentinel-2 GeoTIFF → B4/B3/B2 (RGB) → percentile clip → uint8 →
    PIL → resize 224×224 → ImageNet normalise → DINOv2 → CLS [D] → L2 norm

  SAR:
    Sentinel-1 GeoTIFF → VV/VH (linear) → dB → clip [-30,0] →
    pseudo-RGB [VV|VH|diff] → uint8 → PIL → resize 224×224 →
    ImageNet normalise → DINOv2 → CLS [D] → L2 norm

Usage
─────
  # Full pipeline (spring season, 10K scenes, vitb14)
  python generate_embeddings.py \\
      --root /data/sen12ms \\
      --output-dir ../backend/embeddings \\
      --seasons spring \\
      --max-scenes 10000 \\
      --model dinov2_vitb14 \\
      --batch-size 32 \\
      --device cpu

  # Build FAISS indexes immediately after
  python generate_embeddings.py --root /data/sen12ms --build-index

  # Quick smoke test (100 scenes, tiny model)
  python generate_embeddings.py \\
      --root /data/sen12ms \\
      --max-scenes 100 \\
      --model dinov2_vits14 \\
      --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Resolve path to the backend package ──────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dataset_loader import SEN12MSLoader, SceneRecord, ALL_SEASONS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 embedding dimension per variant
# ─────────────────────────────────────────────────────────────────────────────

_DIM_MAP: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared preprocessing transform (ImageNet stats, 224×224 centre-crop)
# ─────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize(
        224,
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset — wraps SEN12MS scene list for batch DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class SEN12MSEmbeddingDataset(Dataset):
    """
    Returns (optical_tensor, sar_tensor, scene_idx) per scene.

    Both tensors are pre-normalised [3, 224, 224] float32 ready for DINOv2.
    scene_idx is an integer so we can write results back in the right row.
    """

    def __init__(self, scenes: list[SceneRecord], loader: SEN12MSLoader) -> None:
        self._scenes = scenes
        self._loader = loader

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        scene = self._scenes[idx]

        # ── Optical preprocessing ───────────────────────────────────────
        try:
            optical_np = self._loader.load_optical_rgb(scene.optical_path)   # [3,H,W] uint8
            optical_pil = Image.fromarray(
                optical_np.transpose(1, 2, 0),   # HWC for PIL
                mode="RGB",
            )
            optical_tensor = _TRANSFORM(optical_pil)
        except Exception as exc:
            logger.warning("Optical load error at idx %d (%s): %s", idx, scene.scene_id, exc)
            optical_tensor = torch.zeros(3, 224, 224)

        # ── SAR preprocessing ───────────────────────────────────────────
        try:
            sar_tensor = self._preprocess_sar(scene)
        except Exception as exc:
            logger.warning("SAR load error at idx %d (%s): %s", idx, scene.scene_id, exc)
            sar_tensor = torch.zeros(3, 224, 224)

        return optical_tensor, sar_tensor, idx

    def _preprocess_sar(self, scene: SceneRecord) -> torch.Tensor:
        """
        SAR → pseudo-RGB PIL image → normalised tensor.

        Pseudo-RGB channels: [VV_dB, VH_dB, VV_dB - VH_dB]
        Each channel clipped to [-30, 0] dB then normalised to [0, 255] uint8.
        """
        sar_db = self._loader.load_sar(scene.sar_path)   # [2, H, W] dB

        def to_uint8(band: np.ndarray) -> np.ndarray:
            # Map [-30, 0] dB → [0, 255]
            b = np.clip(band, -30.0, 0.0)
            b = (b + 30.0) / 30.0 * 255.0
            return b.astype(np.uint8)

        vv   = to_uint8(sar_db[0])
        vh   = to_uint8(sar_db[1])
        diff = to_uint8(sar_db[0] - sar_db[1])           # polarisation ratio

        pseudo_rgb = np.stack([vv, vh, diff], axis=-1)    # [H, W, 3]
        sar_pil = Image.fromarray(pseudo_rgb, mode="RGB")
        return _TRANSFORM(sar_pil)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding pipeline
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Drives DINOv2 inference over a SEN12MS scene list and saves results.

    Attributes
    ----------
    model_name:   DINOv2 variant identifier (e.g. "dinov2_vitb14").
    device:       torch.device for inference.
    batch_size:   DataLoader batch size.
    num_workers:  DataLoader worker processes (0 = main process only).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.embedding_dim: int = _DIM_MAP.get(model_name, 768)
        self._model: torch.nn.Module | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Download (first run) and load the DINOv2 backbone."""
        logger.info("Loading %s on %s …", self.model_name, self.device)
        t0 = time.perf_counter()

        self._model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=True,
            verbose=False,
        )
        self._model.eval()
        self._model = self._model.to(self.device)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Model loaded  |  dim=%d  |  device=%s  |  %.1f s",
            self.embedding_dim,
            self.device,
            elapsed,
        )

    # ── Main pipeline ──────────────────────────────────────────────────────

    def run(
        self,
        scenes: list[SceneRecord],
        loader: SEN12MSLoader,
        output_dir: Path,
    ) -> None:
        """
        Generate embeddings for all scenes and write output files.

        Args:
            scenes:     Ordered list of SceneRecord (from SEN12MSLoader.discover()).
            loader:     SEN12MSLoader instance (provides load_optical_rgb / load_sar).
            output_dir: Directory where optical.npy, sar.npy, metadata.json are saved.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() before run().")

        n = len(scenes)
        d = self.embedding_dim
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-allocate output arrays
        optical_embs = np.zeros((n, d), dtype=np.float32)
        sar_embs     = np.zeros((n, d), dtype=np.float32)

        dataset = SEN12MSEmbeddingDataset(scenes, loader)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,          # order must match metadata
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        processed = 0
        t_start = time.perf_counter()

        logger.info(
            "Embedding %d scenes  |  batch=%d  |  workers=%d",
            n, self.batch_size, self.num_workers,
        )

        with torch.no_grad():
            for batch_optical, batch_sar, batch_idx in dataloader:
                # Move to device
                batch_optical = batch_optical.to(self.device, non_blocking=True)
                batch_sar     = batch_sar.to(self.device, non_blocking=True)

                # Forward pass — DINOv2 returns CLS token directly
                opt_cls = self._model(batch_optical)   # [B, D]
                sar_cls = self._model(batch_sar)        # [B, D]

                # L2 normalise (in-place on CPU copy)
                opt_np = F.normalize(opt_cls, dim=1).cpu().float().numpy()
                sar_np = F.normalize(sar_cls, dim=1).cpu().float().numpy()

                # Write to pre-allocated arrays using original scene indices
                idx = batch_idx.numpy()
                optical_embs[idx] = opt_np
                sar_embs[idx]     = sar_np

                processed += len(idx)
                self._log_progress(processed, n, t_start)

        # ── Global L2 re-normalise (should already be unit norm, but enforce) ──
        optical_embs = self._l2_normalise(optical_embs)
        sar_embs     = self._l2_normalise(sar_embs)

        # ── Save ─────────────────────────────────────────────────────────────
        self._save(optical_embs, sar_embs, scenes, output_dir)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Done  |  %d scenes  |  %.1f s  |  %.1f scenes/s",
            n, elapsed, n / elapsed,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation; guards against zero-norm rows."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return matrix / norms

    @staticmethod
    def _save(
        optical_embs: np.ndarray,
        sar_embs: np.ndarray,
        scenes: list[SceneRecord],
        output_dir: Path,
    ) -> None:
        """Write optical.npy, sar.npy, and metadata.json to output_dir."""
        opt_path  = output_dir / "optical.npy"
        sar_path  = output_dir / "sar.npy"
        meta_path = output_dir / "metadata.json"

        np.save(str(opt_path), optical_embs)
        logger.info("Saved  %s  %s", opt_path, optical_embs.shape)

        np.save(str(sar_path), sar_embs)
        logger.info("Saved  %s  %s", sar_path, sar_embs.shape)

        metadata = [
            {
                "scene_id":   s.scene_id,
                "lat":        s.lat,
                "lon":        s.lon,
                "season":     s.season,
                "lclu_class": s.lclu_class,
            }
            for s in scenes
        ]
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved  %s  (%d records)", meta_path, len(metadata))

    @staticmethod
    def _log_progress(processed: int, total: int, t_start: float) -> None:
        """Print progress every 250 scenes."""
        if processed % 250 != 0 and processed != total:
            return
        elapsed = time.perf_counter() - t_start
        pct = processed / total * 100
        rate = processed / elapsed if elapsed > 0 else 0
        eta  = (total - processed) / rate if rate > 0 else 0
        print(
            f"\r  [{processed:>6d}/{total}]  {pct:5.1f}%  "
            f"{rate:5.1f} scenes/s  ETA: {eta:5.0f}s",
            end="",
            flush=True,
        )
        if processed == total:
            print()  # final newline


# ─────────────────────────────────────────────────────────────────────────────
# Embedding statistics and sanity check
# ─────────────────────────────────────────────────────────────────────────────

def print_embedding_stats(output_dir: Path) -> None:
    """Print shape, norm, and inter-modal cosine similarity summary."""
    opt_path = output_dir / "optical.npy"
    sar_path = output_dir / "sar.npy"

    if not opt_path.exists() or not sar_path.exists():
        logger.warning("Cannot find embedding files in %s", output_dir)
        return

    opt = np.load(str(opt_path)).astype(np.float32)
    sar = np.load(str(sar_path)).astype(np.float32)

    print("\n── Embedding Statistics ─────────────────────────────────")
    print(f"  optical.npy : {opt.shape}  dtype={opt.dtype}")
    print(f"  sar.npy     : {sar.shape}  dtype={sar.dtype}")

    # L2 norms (should be 1.0 ± 1e-6 after normalisation)
    opt_norms = np.linalg.norm(opt, axis=1)
    sar_norms = np.linalg.norm(sar, axis=1)
    print(f"\n  Optical L2 norm  : mean={opt_norms.mean():.6f}  std={opt_norms.std():.6f}")
    print(f"  SAR     L2 norm  : mean={sar_norms.mean():.6f}  std={sar_norms.std():.6f}")

    # Cross-modal cosine similarity for matched pairs (diagonal)
    n_sample = min(1000, len(opt))
    idx = np.random.choice(len(opt), n_sample, replace=False)
    cos_matched  = (opt[idx] * sar[idx]).sum(axis=1)         # matched pairs
    cos_random   = (opt[idx] * sar[np.random.permutation(idx)]).sum(axis=1)  # random

    print(f"\n  Cross-modal cosine (matched pairs, N={n_sample}):")
    print(f"    mean={cos_matched.mean():.4f}  std={cos_matched.std():.4f}")
    print(f"\n  Cross-modal cosine (random pairs,  N={n_sample}):")
    print(f"    mean={cos_random.mean():.4f}  std={cos_random.std():.4f}")
    print(
        "\n  NOTE: matched ≈ random is expected for zero-shot features.\n"
        "  Cross-modal alignment improves after contrastive fine-tuning."
    )
    print("─────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GeoSense-X — DINOv2 Embedding Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # MVP: 10 000 spring scenes, vitb14, CPU
  python generate_embeddings.py \\
      --root /data/sen12ms \\
      --max-scenes 10000 \\
      --model dinov2_vitb14 \\
      --batch-size 32

  # GPU, all seasons, then build FAISS index
  python generate_embeddings.py \\
      --root /data/sen12ms \\
      --seasons spring summer \\
      --model dinov2_vitb14 \\
      --batch-size 64 \\
      --device cuda \\
      --num-workers 4 \\
      --build-index

  # Quick smoke test
  python generate_embeddings.py \\
      --root /data/sen12ms \\
      --max-scenes 50 \\
      --model dinov2_vits14 \\
      --batch-size 8 \\
      --stats
""",
    )
    p.add_argument(
        "--root", type=str, required=True,
        help="SEN12MS root directory (contains ROIs*/ sub-directories)",
    )
    p.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent.parent / "backend" / "embeddings"),
        help="Where to write optical.npy, sar.npy, metadata.json "
             "(default: ../backend/embeddings)",
    )
    p.add_argument(
        "--model", type=str, default="dinov2_vitb14",
        choices=list(_DIM_MAP.keys()),
        help="DINOv2 variant. vitb14 (768-d) is the recommended default.",
    )
    p.add_argument(
        "--seasons", nargs="+", default=["spring"], choices=ALL_SEASONS,
        help="Seasons to include (default: spring)",
    )
    p.add_argument(
        "--max-scenes", type=int, default=None,
        help="Cap total scene count for quick tests",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32)",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes (default: 0 = main process). "
             "Increase on Linux/Mac for speed.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu | cuda | mps (auto-detected if omitted)",
    )
    p.add_argument(
        "--build-index", action="store_true",
        help="Call build_indexes() after embedding generation",
    )
    p.add_argument(
        "--index-dir", type=str, default=None,
        help="FAISS index output dir (default: ../backend/indexes)",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print embedding statistics after generation",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )

    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)

    # ── 1. Discover scenes ─────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  GeoSense-X — DINOv2 Embedding Generation Pipeline  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    loader = SEN12MSLoader(
        root_dir=args.root,
        seasons=args.seasons,
        max_scenes=args.max_scenes,
    )
    scenes = loader.discover(verbose=True)

    if not scenes:
        logger.error("No scenes found. Check --root and --seasons.")
        sys.exit(1)

    logger.info(
        "Will embed %d scenes  |  seasons=%s  |  model=%s  |  d=%d",
        len(scenes),
        args.seasons,
        args.model,
        _DIM_MAP.get(args.model, 768),
    )

    # Estimate time on CPU
    scenes_per_sec_cpu = 8.0  # approximate for vitb14 on modern CPU
    if args.device != "cuda":
        est_min = len(scenes) / scenes_per_sec_cpu / 60
        logger.info(
            "Estimated time (CPU): %.0f min  "
            "  Tip: use --device cuda or --model dinov2_vits14 to speed up.",
            est_min,
        )

    # ── 2. Load model ──────────────────────────────────────────────────────
    pipeline = EmbeddingPipeline(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pipeline.load_model()

    # ── 3. Generate embeddings ─────────────────────────────────────────────
    pipeline.run(scenes, loader, output_dir)

    # ── 4. Print statistics ────────────────────────────────────────────────
    if args.stats:
        print_embedding_stats(output_dir)

    # ── 5. Optionally build FAISS indexes ──────────────────────────────────
    if args.build_index:
        # Dynamically resolve index_dir relative to the backend
        index_dir = Path(args.index_dir) if args.index_dir else (
            Path(__file__).parent.parent / "backend" / "indexes"
        )
        logger.info("Building FAISS indexes → %s", index_dir)

        # Import from backend
        from services.faiss_service import build_indexes
        build_indexes(output_dir, index_dir)
        logger.info("FAISS indexes ready at %s", index_dir)

    print("\nNext step:")
    if not args.build_index:
        index_dir = Path(args.index_dir) if args.index_dir else (
            Path(__file__).parent.parent / "backend" / "indexes"
        )
        print(f"  python generate_embeddings.py ... --build-index")
        print(f"  OR:")
        print(f"  python -c \"")
        print(f"    from pathlib import Path")
        print(f"    from services.faiss_service import build_indexes")
        print(f"    build_indexes(Path('{output_dir}'), Path('{index_dir}'))\"")
    print("\nThen start the API:")
    print("  cd backend && uvicorn main:app --reload\n")


if __name__ == "__main__":
    main()
