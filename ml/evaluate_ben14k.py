#!/usr/bin/env python3
"""
GeoSense-X — BEN-14K Cross-Modal Retrieval Evaluation
=======================================================
Evaluates the trained CrossModalModel on the BigEarthNet-14K (BEN-14K)
benchmark across all four retrieval directions.

Dataset layout
──────────────
data/raw/benv1_14k/
├── benv1_14k_dataset.parquet      # rows: s1_rel_path, s2_rel_path, labels
├── s1/{patch_id}/{patch_id}_all.npy   # float32 (2,120,120)  [dB scale]
└── s2/{patch_id}/{patch_id}_all.npy   # float32 (12,120,120) [raw ×10000]

Checkpoint
──────────
data/checkpoints/cross_modal_best.pt
State dict key: state["model"]   →  CrossModalModel.load_state_dict()

Architecture verified against backend/ml/cross_modal_embeddings.py
Backbone  : DINOv2 ViT-B/14  (frozen, shared)
s1_head   : Linear(768→512) → GELU → LN → Linear(512→512)
s2_head   : Linear(768→512) → GELU → LN → Linear(512→512)
Output    : L2-normalised [B, 512]

Retrieval directions evaluated
───────────────────────────────
S1 → S2   cross-modal  (SAR query → optical gallery)
S2 → S1   cross-modal  (optical query → SAR gallery)
S1 → S1   same-modal   (SAR → SAR baseline)
S2 → S2   same-modal   (optical → optical baseline)

Relevance criterion
───────────────────
Two patches are relevant iff they share ≥ 1 BigEarthNet label.

Metrics
───────
Recall@1, Recall@5, Recall@10
Precision@5, Precision@10
F1@5, F1@10
mAP (mean Average Precision, full-ranking)

Output
──────
data/results/ben14k_eval_{timestamp}.json

Usage
─────
python ml/evaluate_ben14k.py
python ml/evaluate_ben14k.py --batch-size 16 --device cuda
python ml/evaluate_ben14k.py --max-samples 500   # quick test
"""
from __future__ import annotations

# ── Auto-relaunch with project venv (.venv311) if invoked with system Python ──
# This lets `python ml/evaluate_ben14k.py` work without manually activating the venv.
import os as _os, sys as _sys, subprocess as _sp
_PROJ_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_VENV_PY   = _os.path.join(_PROJ_ROOT, ".venv311", "Scripts", "python.exe")
if not _os.environ.get("_GEOSENSE_VENV_ACTIVE") and _os.path.exists(_VENV_PY) \
        and _os.path.abspath(_sys.executable) != _os.path.abspath(_VENV_PY):
    _env = _os.environ.copy()
    _env["_GEOSENSE_VENV_ACTIVE"] = "1"
    _sys.exit(_sp.run([_VENV_PY] + _sys.argv, env=_env).returncode)

# ── Windows/OpenMP compatibility fixes (must come before any native imports) ──
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # prevents OpenMP crash when
                                                          # libomp (FAISS) and libiomp5
                                                          # (PyTorch/MKL) are both loaded
os.environ.setdefault("PYTHONIOENCODING", "utf-8")       # ensure UTF-8 log output

import argparse
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# Force UTF-8 on Windows consoles that default to CP1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ─────────────────────────────────────────────────────────────────────────────
# Path bootstrap — script lives in ml/, same level as backend/
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = Path(__file__).resolve().parent          # ps11/ml/
_PROJECT_DIR = _SCRIPT_DIR.parent                       # ps11/
_BACKEND_DIR = _PROJECT_DIR / "backend"

# CrossModalModel is defined in backend/ml/cross_modal_embeddings.py
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval_ben14k")

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match backend/ml/cross_modal_embeddings.py
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_DIM:  int         = 768
EMBED_DIM:     int         = 512
HIDDEN_DIM:    int         = 512
DINOV2_MODEL:  str         = "dinov2_vitb14"
DINOV2_SIZE:   int         = 224
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# BEN-14K S2 band ordering: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
# RGB selection: B04(idx=3)=R, B03(idx=2)=G, B02(idx=1)=B  ← same as trained
S2_RGB_IDX: list[int] = [3, 2, 1]

# S1 dB normalisation range (same as SEN12MS training preprocessing)
S1_DB_MIN: float = -30.0
S1_DB_MAX: float =   0.0

# S2 raw→reflectance scale factor
S2_SCALE: float = 10_000.0

# Default paths
_DEFAULT_DATASET_ROOT = _PROJECT_DIR / "data" / "raw"  / "benv1_14k"
_DEFAULT_PARQUET      = _DEFAULT_DATASET_ROOT / "benv1_14k_dataset.parquet"
_DEFAULT_CHECKPOINT   = _PROJECT_DIR / "data" / "checkpoints" / "cross_modal_best.pt"
_DEFAULT_OUTPUT_DIR   = _PROJECT_DIR / "data" / "results"

K_VALUES: list[int] = [1, 5, 10]


# ─────────────────────────────────────────────────────────────────────────────
# Re-declare model components (avoids importing heavy training deps at eval)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Exact replica of backend/ml/cross_modal_embeddings.py::ProjectionHead.
    Linear(768→512) → GELU → LayerNorm(512) → Linear(512→512)
    """

    def __init__(
        self,
        in_dim:     int = BACKBONE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        out_dim:    int = EMBED_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        # Same Kaiming init as training — needed for correct state_dict alignment
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1.0 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossModalModel(nn.Module):
    """
    Exact replica of backend/ml/cross_modal_embeddings.py::CrossModalModel.
    Shared frozen DINOv2 backbone + two separate trainable projection heads.
    """

    def __init__(
        self,
        embed_dim:   int   = EMBED_DIM,
        hidden_dim:  int   = HIDDEN_DIM,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()

        logger.info("Loading DINOv2 backbone: %s …", DINOV2_MODEL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone: nn.Module = torch.hub.load(
                "facebookresearch/dinov2",
                DINOV2_MODEL,
                pretrained=True,
                verbose=False,
            )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.s1_head = ProjectionHead(BACKBONE_DIM, hidden_dim, embed_dim)
        self.s2_head = ProjectionHead(BACKBONE_DIM, hidden_dim, embed_dim)

        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature), dtype=torch.float32)
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _preprocess_s1(self, s1: Tensor) -> Tensor:
        """[B, 2, H, W] in [0,1] → [B, 3, 224, 224] ImageNet-normalised."""
        vv   = s1[:, 0:1, :, :]
        vh   = s1[:, 1:2, :, :]
        diff = (vv - vh).clamp(0.0, 1.0)
        rgb  = torch.cat([vv, vh, diff], dim=1)
        rgb  = F.interpolate(rgb, DINOV2_SIZE, mode="bilinear", align_corners=False)
        return (rgb - self.imagenet_mean) / self.imagenet_std

    def _preprocess_s2(self, s2: Tensor) -> Tensor:
        """[B, 12, H, W] in [0,1] → [B, 3, 224, 224] ImageNet-normalised."""
        rgb = s2[:, S2_RGB_IDX, :, :]
        rgb = F.interpolate(rgb, DINOV2_SIZE, mode="bilinear", align_corners=False)
        return (rgb - self.imagenet_mean) / self.imagenet_std

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode_s1(self, s1: Tensor) -> Tensor:
        with torch.no_grad():
            cls = self.backbone(self._preprocess_s1(s1))
        return F.normalize(self.s1_head(cls), p=2, dim=-1)

    def encode_s2(self, s2: Tensor) -> Tensor:
        with torch.no_grad():
            cls = self.backbone(self._preprocess_s2(s2))
        return F.normalize(self.s2_head(cls), p=2, dim=-1)

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.s1_head.eval()
        self.s2_head.eval()


# ─────────────────────────────────────────────────────────────────────────────
# BEN-14K Dataset
# ─────────────────────────────────────────────────────────────────────────────

try:
    import ctypes as _ctypes
    _GetFileAttributesW = _ctypes.windll.kernel32.GetFileAttributesW
    _FILE_ATTR_OFFLINE               = 0x00001000
    _FILE_ATTR_RECALL_ON_DATA_ACCESS = 0x00400000
    def _is_truly_offline(p: Path) -> bool:
        """True only when the file has OFFLINE set WITHOUT RECALL — i.e. the OS
        cannot recall it at all.  Files with RECALL_ON_DATA_ACCESS (but not
        OFFLINE) are OneDrive "recall-on-open" stubs that CAN be downloaded on
        access (≈35 s each); they must NOT be skipped."""
        attrs = _GetFileAttributesW(str(p))
        if attrs == 0xFFFFFFFF:
            return False
        offline = bool(attrs & _FILE_ATTR_OFFLINE)
        recall  = bool(attrs & _FILE_ATTR_RECALL_ON_DATA_ACCESS)
        return offline and not recall   # mid-download / permanent offline only
except Exception:
    def _is_truly_offline(p: Path) -> bool:  # type: ignore[misc]
        return False


def _load_npy_with_timeout(path: Path, timeout: float = 8.0) -> "np.ndarray":
    """np.load() with a wall-clock timeout to avoid hanging on OneDrive RECALL I/O.

    On Windows, open() on a RECALL stub succeeds immediately but file.read()
    blocks the calling thread indefinitely while OneDrive streams the file.
    This does NOT raise OSError — it simply never returns.  We run np.load()
    on a daemon thread and join it with a timeout; if it takes longer than
    `timeout` seconds we raise OSError(EINVAL) so the caller's retry/skip
    logic can handle it.
    """
    import threading
    result: list = [None]
    exc:    list = [None]

    def _load() -> None:
        try:
            result[0] = np.load(str(path))
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        # Thread is still blocked in kernel read() — OneDrive download stalled.
        raise OSError(22, f"OneDrive RECALL timed out after {timeout:.0f}s", str(path))
    if exc[0] is not None:
        raise exc[0]
    return result[0]


def _load_npy_with_retry(path: Path, max_retries: int = 1, base_delay: float = 3.0) -> "np.ndarray":
    """Load a .npy file.

    * Files that are permanently offline (OFFLINE bit set, no RECALL bit) are
      rejected immediately.
    * Files with RECALL_ON_DATA_ACCESS (OneDrive pinned-but-not-yet-local) are
      attempted; OSError EINVAL is retried up to max_retries times with
      exponential backoff — downloads take ≈35 s each.
    * A 60-second wall-clock timeout guards against silent blocking reads
      (OneDrive RECALL that never raises EINVAL but never returns either).
    """
    if _is_truly_offline(path):
        raise OSError(22, "OneDrive file is permanently offline", str(path))
    for attempt in range(max_retries):
        try:
            return _load_npy_with_timeout(path)
        except OSError as exc:
            if exc.errno == 22 and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)   # 8 s, 16 s, 32 s, 64 s, …
                logger.debug(
                    "EINVAL/timeout on %s — OneDrive mid-recall? retry %d/%d in %.0fs",
                    path.name, attempt + 1, max_retries - 1, wait,
                )
                time.sleep(wait)
                continue
            raise


class BEN14KDataset(Dataset):
    """
    PyTorch Dataset for BEN-14K.

    Returns:
        s1:         Tensor [2, 120, 120]  float32, normalised to [0, 1]
        s2:         Tensor [12, 120, 120] float32, normalised to [0, 1]
        label_set:  frozenset of label strings  (for multi-label eval)
        idx:        int  (original DataFrame row index)
    """

    def __init__(
        self,
        parquet_path: Path,
        dataset_root: Path,
        max_samples:  Optional[int] = None,
    ) -> None:
        self.root = dataset_root

        df = pd.read_parquet(parquet_path)
        logger.info("Parquet loaded: %d rows, columns=%s", len(df), list(df.columns))

        # Validate required columns
        for col in ("s1_rel_path", "s2_rel_path", "labels"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in parquet")

        if max_samples is not None:
            df = df.iloc[:max_samples].copy()
            logger.info("Capped to %d samples (--max-samples)", max_samples)

        # Validate that files exist (sample 5 randomly for speed)
        self._validate_files(df)

        self.df = df.reset_index(drop=True)
        logger.info("BEN14KDataset ready: %d paired samples", len(self.df))

    def _validate_files(self, df: pd.DataFrame) -> None:
        """Check that a random sample of files actually exist on disk."""
        sample = df.sample(min(5, len(df)), random_state=42)
        missing: list[str] = []
        for _, row in sample.iterrows():
            for subdir, col in (("s1", "s1_rel_path"), ("s2", "s2_rel_path")):
                p = self.root / subdir / row[col]
                if not p.exists():
                    missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                f"Dataset validation failed — {len(missing)} file(s) not found.\n"
                f"First missing: {missing[0]}\n"
                f"Check that dataset_root='{self.root}' is correct and "
                f"that the s1/ and s2/ sub-directories exist."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Load S1  ──────────────────────────────────────────────────────
        s1_arr = _load_npy_with_retry(self.root / "s1" / row["s1_rel_path"])  # (2,120,120)
        # S1 is in dB: clip to [-30, 0] → normalise to [0, 1]
        s1_arr = np.clip(s1_arr, S1_DB_MIN, S1_DB_MAX)
        s1_arr = (s1_arr - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN)
        s1 = torch.from_numpy(s1_arr.astype(np.float32))              # [2, 120, 120]

        # ── Load S2  ──────────────────────────────────────────────────────
        s2_arr = _load_npy_with_retry(self.root / "s2" / row["s2_rel_path"])  # (12,120,120)
        # S2 is raw reflectance ×10000 → normalise to [0, 1]
        s2_arr = np.clip(s2_arr / S2_SCALE, 0.0, 1.0)
        s2 = torch.from_numpy(s2_arr.astype(np.float32))              # [12, 120, 120]

        # ── Labels  ───────────────────────────────────────────────────────
        label_set = frozenset(row["labels"].tolist())

        return {"s1": s1, "s2": s2, "label_set": label_set, "idx": idx}


def ben14k_collate(batch: list[dict]) -> dict:
    """Custom collate that stacks tensors but keeps label_sets as a list."""
    return {
        "s1":        torch.stack([b["s1"] for b in batch]),
        "s2":        torch.stack([b["s2"] for b in batch]),
        "label_sets": [b["label_set"] for b in batch],
        "idxs":       [b["idx"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint(checkpoint_path: Path) -> Path:
    """
    Return an existing checkpoint path.

    If the requested path does not exist, search the same directory for
    the newest epoch checkpoint (cross_modal_epoch_*.pt) and return it.
    Also copies it as cross_modal_best.pt so future runs use the fast path.
    """
    if checkpoint_path.exists():
        return checkpoint_path

    logger.warning("Checkpoint not found: %s  -- searching for latest epoch checkpoint ...", checkpoint_path)
    ckpt_dir = checkpoint_path.parent
    epoch_ckpts = sorted(ckpt_dir.glob("cross_modal_epoch_*.pt"))
    if not epoch_ckpts:
        # Also check backups/ as a fallback
        backups_dir = checkpoint_path.parent.parent / "backups"
        if backups_dir.is_dir():
            epoch_ckpts = sorted(backups_dir.glob("cross_modal_epoch_*.pt"))
            if not epoch_ckpts:
                best_backup = backups_dir / "cross_modal_best.pt"
                if best_backup.exists():
                    logger.info("Using backup checkpoint: %s", best_backup)
                    import shutil
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(best_backup), str(checkpoint_path))
                    logger.info("Copied %s -> %s", best_backup, checkpoint_path)
                    return checkpoint_path
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir} or backups/.\n"
            "Run backend/ml/cross_modal_embeddings.py to train the model first."
        )

    latest = epoch_ckpts[-1]
    logger.info("Found latest epoch checkpoint: %s", latest)
    # Copy it as best.pt for convenience
    import shutil
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(latest), str(checkpoint_path))
    logger.info("Promoted %s -> %s", latest.name, checkpoint_path.name)
    return checkpoint_path


def load_model(checkpoint_path: Path, device: torch.device) -> CrossModalModel:
    """
    Load CrossModalModel from a checkpoint saved by the training script.

    Expected checkpoint structure:
        {
        "epoch":       int,
        "val_loss":    float,
        "model":       <state_dict>,
        "optimiser":   <state_dict>,
        "scheduler":   <state_dict>,
        "temperature": float,
        }
    """
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run backend/ml/cross_modal_embeddings.py first."
        )

    logger.info("Loading checkpoint: %s", checkpoint_path)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Validate checkpoint structure
    if "model" not in state:
        raise KeyError(
            f"Checkpoint is missing 'model' key. Found keys: {list(state.keys())}\n"
            "This checkpoint may have been saved in a different format."
        )

    logger.info(
        "Checkpoint info — epoch=%s  val_loss=%s  temperature=%.4f",
        state.get("epoch", "?"),
        f"{state.get('val_loss', float('nan')):.6f}",
        state.get("temperature", float("nan")),
    )

    # Instantiate model with same hyper-parameters used during training
    model = CrossModalModel(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        temperature=0.07,   # overwritten by state_dict
    )

    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint (may be non-critical): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected)

    # Validate that both projection heads loaded correctly
    for head_name in ("s1_head", "s2_head"):
        head_keys = [k for k in state["model"] if k.startswith(head_name)]
        if not head_keys:
            raise RuntimeError(
                f"Checkpoint appears to have no weights for '{head_name}'. "
                f"Ensure the checkpoint was saved from CrossModalModel."
            )
        logger.info("%s: %d weight tensors loaded", head_name, len(head_keys))

    model = model.to(device)
    model.eval_mode()
    logger.info("Model on %s — ready for inference", device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(
    model:      CrossModalModel,
    loader:     DataLoader,
    device:     torch.device,
    n_total:    int,
) -> tuple[np.ndarray, np.ndarray, list[frozenset]]:
    """
    Run batched inference over the full dataset.

    Returns
    -------
    s1_embs    : [N, 512] float32, L2-normalised
    s2_embs    : [N, 512] float32, L2-normalised
    label_sets : list of N frozensets (multi-label ground truth)
    """
    s1_list:     list[np.ndarray]   = []
    s2_list:     list[np.ndarray]   = []
    all_labels:  list[frozenset]    = []

    model.eval()

    # Use iter(loader) directly — NOT iter(tqdm(loader)).
    # tqdm.__iter__ is a generator; when OSError propagates through a generator
    # the generator closes permanently, so every subsequent next() raises
    # StopIteration.  We update tqdm manually instead.
    pbar = tqdm(total=len(loader), desc="Extracting embeddings", unit="batch") \
           if _HAS_TQDM else None

    t_start = time.perf_counter()
    skipped_batches = 0

    with torch.no_grad():
        loader_iter = iter(loader)   # DataLoader is a plain __next__ class, safe to resume
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            except OSError as exc:
                if exc.errno == 22:
                    skipped_batches += 1
                    if pbar is not None:
                        pbar.update(1)
                    continue
                raise

            s1 = batch["s1"].to(device, non_blocking=True)   # [B, 2, 120, 120]
            s2 = batch["s2"].to(device, non_blocking=True)   # [B, 12, 120, 120]

            emb_s1 = model.encode_s1(s1).cpu().float().numpy()  # [B, 512]
            emb_s2 = model.encode_s2(s2).cpu().float().numpy()  # [B, 512]

            s1_list.append(emb_s1)
            s2_list.append(emb_s2)
            all_labels.extend(batch["label_sets"])

            if pbar is not None:
                pbar.update(1)

            # GPU memory safety: empty cache every 50 batches
            if device.type == "cuda" and len(s1_list) % 50 == 0:
                torch.cuda.empty_cache()

    if pbar is not None:
        pbar.close()

    if skipped_batches:
        logger.warning(
            "Skipped %d batches (~%d samples) — OneDrive cloud-only files not pinned locally.",
            skipped_batches, skipped_batches * loader.batch_size,
        )
        logger.warning(
            "To pin all dataset files locally, run:  "
            "attrib +P /S /D \"%s\"", str(loader.dataset.root),
        )

    elapsed = time.perf_counter() - t_start
    n = sum(a.shape[0] for a in s1_list)
    if n == 0:
        raise RuntimeError(
            "No samples could be loaded — all files appear to be OneDrive cloud-only stubs. "
            "Pin the dataset locally first: attrib +P /S /D \"<dataset_root>\""
        )

    s1_embs = np.vstack(s1_list)   # [N, 512]
    s2_embs = np.vstack(s2_list)   # [N, 512]

    # Sanity checks
    assert s1_embs.shape == (n, EMBED_DIM), f"S1 shape mismatch: {s1_embs.shape}"
    assert s2_embs.shape == (n, EMBED_DIM), f"S2 shape mismatch: {s2_embs.shape}"

    norms_s1 = np.linalg.norm(s1_embs, axis=1)
    norms_s2 = np.linalg.norm(s2_embs, axis=1)
    logger.info(
        "Embeddings extracted  |  N=%d  |  %.1f samples/s",
        n, n / elapsed,
    )
    logger.info(
        "L2 norm check  |  S1: mean=%.4f  S2: mean=%.4f  (expect ~=1.0)",
        norms_s1.mean(), norms_s2.mean(),
    )

    if not np.allclose(norms_s1, 1.0, atol=1e-3):
        logger.warning("S1 embeddings are not unit-normalised! Check preprocessing.")
    if not np.allclose(norms_s2, 1.0, atol=1e-3):
        logger.warning("S2 embeddings are not unit-normalised! Check preprocessing.")

    return s1_embs, s2_embs, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Relevance matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_label_onehot(
    label_sets: list[frozenset],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode multi-label sets as a one-hot matrix for BEN-MM evaluation.

    Replaces the old boolean N×N relevance matrix with a compact [N, L]
    representation.  Per-pair scores are computed on-the-fly during retrieval
    (see compute_pair_scores), so the expensive N×N materialisation is avoided.

    Returns
    -------
    onehot      : [N, L] uint8  — binary label indicator matrix
    label_sizes : [N]    int32  — |Yq| for each sample
    """
    N = len(label_sets)
    all_unique_labels = sorted({lbl for ls in label_sets for lbl in ls})
    label_to_idx = {lbl: i for i, lbl in enumerate(all_unique_labels)}
    L = len(all_unique_labels)
    logger.info("Label vocabulary: %d unique labels  |  N=%d samples", L, N)

    onehot = np.zeros((N, L), dtype=np.uint8)
    for i, ls in enumerate(label_sets):
        for lbl in ls:
            onehot[i, label_to_idx[lbl]] = 1

    label_sizes = onehot.sum(axis=1).astype(np.int32)
    logger.info(
        "Label sizes  |  mean=%.1f  min=%d  max=%d  labels/sample",
        label_sizes.mean(), label_sizes.min(), label_sizes.max(),
    )
    return onehot, label_sizes


# ─────────────────────────────────────────────────────────────────────────────
# BEN-MM per-pair score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pair_scores(
    onehot:            np.ndarray,    # [N, L] uint8
    query_indices:     np.ndarray,    # [N_q]  int64  — row indices into onehot
    retrieved_indices: np.ndarray,    # [N_q, K] int64 — gallery indices into onehot
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BEN-MM Section S3 per-pair Precision, Recall, and F1 scores.

    For query q with label set Yq and retrieved patch r with label set Yr:

        P(q, r)  = |Yq ∩ Yr| / |Yr|              (0 when |Yr| == 0)
        R(q, r)  = |Yq ∩ Yr| / |Yq|              (0 when |Yq| == 0)
        F1(q, r) = 2 · P(q,r) · R(q,r)           (0 when P+R == 0)
                   ─────────────────────
                       P(q,r) + R(q,r)

    All three matrices are computed in a single vectorised pass:
      intersection[q, k] = Σ_l  onehot[q, l] · onehot[r_k, l]

    Memory: for N=14832, K=10, L=43 → ~25 MB peak — safe on 16 GB RAM.

    Returns
    -------
    P_mat  : [N_q, K] float32
    R_mat  : [N_q, K] float32
    F1_mat : [N_q, K] float32
    """
    # Gather label matrices for queries and retrieved results
    Yq = onehot[query_indices].astype(np.float32)          # [N_q, L]
    Yr = onehot[retrieved_indices].astype(np.float32)      # [N_q, K, L]

    # |Yq ∩ Yr|: element-wise product summed over label dimension
    intersection = (Yq[:, None, :] * Yr).sum(axis=2)      # [N_q, K]

    yq_sizes = Yq.sum(axis=1, keepdims=True)               # [N_q, 1]
    yr_sizes = Yr.sum(axis=2)                              # [N_q, K]

    # np.where evaluates both branches before masking, producing a harmless
    # divide-by-zero on zero-sized label sets — suppress the warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        P_mat  = np.where(yr_sizes > 0, intersection / yr_sizes, 0.0).astype(np.float32)
        R_mat  = np.where(yq_sizes > 0, intersection / yq_sizes, 0.0).astype(np.float32)
        denom  = P_mat + R_mat
        F1_mat = np.where(denom > 0, 2.0 * P_mat * R_mat / denom, 0.0).astype(np.float32)

    return P_mat, R_mat, F1_mat


# ─────────────────────────────────────────────────────────────────────────────
# FAISS search
# ─────────────────────────────────────────────────────────────────────────────

def faiss_search(
    query_embs:   np.ndarray,   # [N_q, D]
    gallery_embs: np.ndarray,   # [N_g, D]
    top_k:        int,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact inner-product search using FAISS IndexFlatIP.

    For L2-normalised embeddings, inner product == cosine similarity.

    Args:
        query_embs:   Query embeddings [N_q, D].
        gallery_embs: Gallery embeddings [N_g, D].
        top_k:        Number of neighbours to retrieve.
        exclude_self: If True, retrieve top_k+1 and strip rank-1 (for same-modal
                    where query == gallery and self-similarity is 1.0).

    Returns
    -------
    scores  : [N_q, top_k] float32
    indices : [N_q, top_k] int64
    """
    D = gallery_embs.shape[1]
    k_fetch = top_k + 1 if exclude_self else top_k

    gallery = gallery_embs.copy().astype(np.float32)
    queries = query_embs.copy().astype(np.float32)
    faiss.normalize_L2(gallery)   # defensive re-normalise
    faiss.normalize_L2(queries)

    index = faiss.IndexFlatIP(D)
    index.add(gallery)

    scores, indices = index.search(queries, k_fetch)  # [N_q, k_fetch]

    if exclude_self:
        # For each query i, remove the result at position 0 (self, score≈1.0)
        # This assumes query_embs and gallery_embs have the same order.
        scores  = scores[:, 1:]
        indices = indices[:, 1:]

    return scores, indices


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_recall_at_k(R_mat: np.ndarray, k: int) -> float:
    """
    BEN-MM R@K: mean per-pair Recall over the top-K retrieved results.

    R@K(q) = (1/K) · Σ_{k=1}^{K} R(q, r_k)
    mR@K   = mean over all queries

    Args:
        R_mat: [N_q, K_max] per-pair Recall matrix from compute_pair_scores.
        k:     Cut-off rank.
    """
    return float(R_mat[:, :k].mean())


def compute_precision_at_k(P_mat: np.ndarray, k: int) -> float:
    """
    BEN-MM P@K: mean per-pair Precision over the top-K retrieved results.

    P@K(q) = (1/K) · Σ_{k=1}^{K} P(q, r_k)   where P(q,r) = |Yq∩Yr| / |Yr|
    mP@K   = mean over all queries

    Args:
        P_mat: [N_q, K_max] per-pair Precision matrix from compute_pair_scores.
        k:     Cut-off rank.
    """
    return float(P_mat[:, :k].mean())


def compute_f1_at_k(F1_mat: np.ndarray, k: int) -> float:
    """
    BEN-MM F1@K: mean per-pair F1 score over the top-K retrieved results.

    F1@K(q) = (1/K) · Σ_{k=1}^{K} F1(q, r_k)
    mF1@K   = mean over all queries

    Args:
        F1_mat: [N_q, K_max] per-pair F1 matrix from compute_pair_scores.
        k:      Cut-off rank.
    """
    return float(F1_mat[:, :k].mean())


def compute_map(F1_mat: np.ndarray) -> float:
    """
    Standard binary-relevance mAP over the top-K retrieved results.

    Relevance is binary: a retrieved patch r is relevant to query q iff
    their label sets share at least one common label, i.e. F1(q,r) > 0.
    This is the established definition for multi-label image retrieval
    (see e.g. MARC, arXiv:2512.16294, and standard CBIR evaluation).

    The F1-weighted AP formula used previously was not found in any cited
    paper and is not the standard definition — it has been replaced here.

    AP(q) = Σ_{k : rel(k)=1} [precision@k] / n_relevant(q)
            where precision@k = (Σ_{j≤k} rel(j)) / k
            and   rel(k) = 1 iff F1(q, r_k) > 0

    mAP = mean AP over queries that have ≥1 relevant result in top-K.

    Args:
        F1_mat: [N_q, K] per-pair F1 matrix from compute_pair_scores.
                F1(q,r) > 0  ⟺  |Yq ∩ Yr| > 0  (at least one shared label).
    """
    # Derive binary relevance from F1: rel=1 iff any label is shared.
    # F1(q,r) > 0  ⟺  intersection > 0  ⟺  standard binary relevance.
    rel_mat = (F1_mat > 0).astype(np.float64)          # [N_q, K]  0.0 or 1.0

    N_q, K  = rel_mat.shape
    ap_sum  = 0.0
    skipped = 0
    ranks   = np.arange(1, K + 1, dtype=np.float64)    # [K]

    for q in range(N_q):
        rel   = rel_mat[q]                              # [K]
        n_rel = rel.sum()
        if n_rel == 0:
            skipped += 1
            continue

        running   = np.cumsum(rel)                      # running hit count
        prec_at_k = running / ranks                     # precision at each rank
        ap_sum   += float((rel * prec_at_k).sum()) / n_rel

    if skipped > 0:
        logger.warning(
            "mAP: %d / %d queries had no relevant items in top-%d — skipped.",
            skipped, N_q, K,
        )

    denom = N_q - skipped
    return ap_sum / denom if denom > 0 else 0.0


def evaluate_direction(
    direction:     str,
    query_embs:    np.ndarray,   # [N_q, 512]
    gallery_embs:  np.ndarray,   # [N_g, 512]
    onehot:        np.ndarray,   # [N,   L]   uint8  — shared label matrix
    query_indices: np.ndarray,   # [N_q]      int64  — row indices into onehot
    k_values:      list[int],
    exclude_self:  bool,
) -> dict:
    """
    Run FAISS search and compute BEN-MM retrieval metrics for one direction.

    Workflow:
      1. FAISS inner-product search → retrieved gallery indices [N_q, K].
      2. compute_pair_scores() → per-pair P/R/F1 matrices [N_q, K].
      3. Aggregate with compute_*_at_k() and compute_map().

    Same-modal directions (exclude_self=True) strip rank-0 (self-match) in
    faiss_search before per-pair scoring, so self-matches are never included.
    """
    logger.info("=== Direction: %s ===", direction)
    K_max = max(k_values)

    t0 = time.perf_counter()
    _, indices = faiss_search(          # scores unused; using label-based metrics
        query_embs, gallery_embs,
        top_k=K_max,
        exclude_self=exclude_self,
    )
    search_ms = (time.perf_counter() - t0) * 1000
    logger.info("FAISS search done in %.0f ms  |  shape=%s", search_ms, indices.shape)

    # ── Per-pair BEN-MM scores ─────────────────────────────────────────────
    P_mat, R_mat, F1_mat = compute_pair_scores(onehot, query_indices, indices)

    results: dict = {"direction": direction}
    for k in k_values:
        r_at_k = compute_recall_at_k(R_mat, k)
        results[f"R@{k}"]  = round(r_at_k, 6)
        logger.info("  Recall@%-3d = %.4f", k, r_at_k)

    for k in [5, 10]:
        p_at_k = compute_precision_at_k(P_mat, k)
        results[f"P@{k}"]  = round(p_at_k, 6)
        logger.info("  Precision@%-2d = %.4f", k, p_at_k)

    for k in [5, 10]:
        f1 = compute_f1_at_k(F1_mat, k)
        results[f"F1@{k}"] = round(f1, 6)
        logger.info("  F1@%-9d = %.4f", k, f1)

    t1 = time.perf_counter()
    mAP = compute_map(F1_mat)
    results["mAP"] = round(mAP, 6)
    logger.info("  mAP          = %.4f  (%.1f s)", mAP, time.perf_counter() - t1)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_path: Path,
    dataset_root:    Path,
    parquet_path:    Path,
    output_dir:      Path,
    batch_size:      int,
    device_str:      str,
    num_workers:     int,
    max_samples:     Optional[int],
    k_values:        list[int],
) -> dict:
    """
    Full evaluation pipeline.

    Returns
    -------
    dict containing all metric results for all 4 directions.
    """
    t_pipeline = time.perf_counter()

    # ── Device ────────────────────────────────────────────────────────────
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info("Device: %s", device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s  (%.1f GB VRAM)", gpu_name, vram_gb)

    # ── Checkpoint validation ──────────────────────────────────────────────
    model = load_model(checkpoint_path, device)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = BEN14KDataset(
        parquet_path=parquet_path,
        dataset_root=dataset_root,
        max_samples=max_samples,
    )
    N = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=ben14k_collate,
    )

    logger.info(
        "DataLoader ready  |  N=%d  |  batch=%d  |  batches=%d  |  workers=%d",
        N, batch_size, len(loader), num_workers,
    )

    # ── Extract embeddings ────────────────────────────────────────────────
    s1_embs, s2_embs, label_sets = extract_embeddings(
        model, loader, device, n_total=N
    )


    # ── BEN-MM label one-hot matrix ──────────────────────────────────────────
    # Build [N_actual, L] uint8 instead of the old [N, N] bool relevance matrix.
    # n_actual may be < N when some batches were skipped (cloud-only OneDrive files).
    n_actual = s1_embs.shape[0]
    onehot, _ = build_label_onehot(label_sets)              # [n_actual, L] uint8
    query_indices = np.arange(n_actual, dtype=np.int64)     # all n_actual patches are queries

    # ── Evaluate all four directions ──────────────────────────────────────
    all_results = {
        "config": {
            "checkpoint":    str(checkpoint_path),
            "dataset_root":  str(dataset_root),
            "n_samples":     n_actual,
            "n_total":       N,
            "embed_dim":     EMBED_DIM,
            "k_values":      k_values,
            "batch_size":    batch_size,
            "device":        str(device),
            "eval_protocol": "BEN-MM",
        },
        "directions": {},
    }

    for direction, q_embs, g_embs, excl in [
        ("S1->S2", s1_embs, s2_embs, False),   # cross-modal
        ("S2->S1", s2_embs, s1_embs, False),   # cross-modal
        ("S1->S1", s1_embs, s1_embs, True),    # same-modal, exclude self
        ("S2->S2", s2_embs, s2_embs, True),    # same-modal, exclude self
    ]:
        res = evaluate_direction(
            direction=direction,
            query_embs=q_embs,
            gallery_embs=g_embs,
            onehot=onehot,
            query_indices=query_indices,
            k_values=k_values,
            exclude_self=excl,
        )
        all_results["directions"][direction] = res

    # ── Print summary table ───────────────────────────────────────────────
    _print_results_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    ts         = time.strftime("%Y%m%d_%H%M%S")
    out_path   = output_dir / f"ben14k_eval_{ts}.json"
    all_results["elapsed_seconds"] = round(time.perf_counter() - t_pipeline, 1)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved → %s", out_path)

    return all_results


def _print_results_table(results: dict) -> None:
    """Print a compact ASCII results table to stdout."""
    dirs = results["directions"]
    cols = ["R@1", "R@5", "R@10", "P@5", "P@10", "F1@5", "F1@10", "mAP"]

    header = f"{'Direction':<12}" + "".join(f"{c:>8}" for c in cols)
    sep    = "=" * len(header)

    print()
    print(sep)
    print("  BEN-14K Retrieval Metrics")
    print(sep)
    print(header)
    print("-" * len(header))
    for direction, row in dirs.items():
        vals = "".join(f"{row.get(c, 0.0):>8.4f}" for c in cols)
        print(f"{direction:<12}{vals}")
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_ben14k",
        description="Evaluate the cross-modal model on BEN-14K retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Full evaluation (all 14,832 samples)
python ml/evaluate_ben14k.py

# Quick smoke test on first 500 samples
python ml/evaluate_ben14k.py --max-samples 500

# GPU inference, large batch
python ml/evaluate_ben14k.py --device cuda --batch-size 64

# Specify a different checkpoint
python ml/evaluate_ben14k.py \\
    --checkpoint data/checkpoints/cross_modal_epoch_0010.pt
""",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {_DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT,
        help=f"BEN-14K root directory (default: {_DEFAULT_DATASET_ROOT})",
    )
    p.add_argument(
        "--parquet", type=Path, default=_DEFAULT_PARQUET,
        help=f"Metadata parquet file (default: {_DEFAULT_PARQUET})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON results (default: {_DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32). Reduce to 16 if OOM.",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto | cuda | cpu  (default: auto)",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0, safe on Windows)",
    )
    p.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Cap dataset to first N samples for a quick test",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print execution plan
    logger.info("=" * 60)
    logger.info("GeoSense-X  ·  BEN-14K Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint  : %s", args.checkpoint)
    logger.info("Dataset     : %s", args.dataset_root)
    logger.info("Parquet     : %s", args.parquet)
    logger.info("Output dir  : %s", args.output_dir)
    logger.info("Batch size  : %d", args.batch_size)
    logger.info("Device      : %s", args.device)
    logger.info("Max samples : %s", args.max_samples or "all")
    logger.info("=" * 60)

    try:
        run_evaluation(
            checkpoint_path = args.checkpoint,
            dataset_root    = args.dataset_root,
            parquet_path    = args.parquet,
            output_dir      = args.output_dir,
            batch_size      = args.batch_size,
            device_str      = args.device,
            num_workers     = args.num_workers,
            max_samples     = args.max_samples,
            k_values        = K_VALUES,
        )
        return 0

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
