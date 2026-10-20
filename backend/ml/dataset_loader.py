"""
GeoSense-X — SEN12MS PyTorch Dataset Loader
=============================================
Provides ``SEN12MSDataset``, a PyTorch Dataset that reads paired
Sentinel-1 / Sentinel-2 / Land-Cover TIFF triplets from the SEN12MS
dataset using the metadata CSV produced by ``generate_metadata.py``.

Measured file specs (ROIs2017_winter)
──────────────────────────────────────
  Sentinel-1  (SAR)   :  2  bands × 256 × 256, float32, dB scale
                          Band 0 = VV  [-30,  0] dB
                          Band 1 = VH  [-30,  0] dB
  Sentinel-2  (Optical): 13 bands × 256 × 256, uint16
                          Reflectance × 10 000 (range ≈ 0 – 10 000)
  Land Cover  (Labels) :  4 bands × 256 × 256, uint8
                          Band 0 = IGBP (17 classes, primary label)
                          Band 1 = UMD  scheme
                          Band 2 = LAI  scheme
                          Band 3 = BGC  scheme

Normalization applied by default
──────────────────────────────────
  S1 :  clip to [S1_MIN, S1_MAX] dB → scale to [0, 1]
  S2 :  divide by 10 000, clip to [0, 1]
  LC :  no normalisation — returned as int64 class indices

Return dict per sample
───────────────────────
  {
      "patch_key" : str,          # e.g. "47_p729"
      "region_id" : int,          # e.g. 47
      "s1"        : Tensor,       # [2,  256, 256] float32
      "s2"        : Tensor,       # [13, 256, 256] float32
      "lc"        : Tensor,       # [4,  256, 256] int64
  }

Usage
─────
  from backend.ml.dataset_loader import SEN12MSDataset, create_dataloader

  ds     = SEN12MSDataset("data/metadata/train.csv", "data/raw")
  loader = create_dataloader(ds, batch_size=8, shuffle=True, num_workers=4)

  for batch in loader:
      s1 = batch["s1"]   # [8,  2, 256, 256]
      s2 = batch["s2"]   # [8, 13, 256, 256]
      lc = batch["lc"]   # [8,  4, 256, 256]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation constants  (empirical, measured on ROIs2017_winter)
# ─────────────────────────────────────────────────────────────────────────────

# S1 dB clipping range — values outside are clamped before scaling to [0, 1]
S1_MIN: float = -30.0
S1_MAX: float =   0.0

# S2 reflectance scale factor (raw uint16 = reflectance × S2_SCALE)
S2_SCALE: float = 10_000.0

# Expected spatial dimensions of every patch
PATCH_H: int = 256
PATCH_W: int = 256

# Expected band counts
S1_BANDS: int =  2
S2_BANDS: int = 13
LC_BANDS: int =  4


# ─────────────────────────────────────────────────────────────────────────────
# SEN12MS Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SEN12MSDataset(Dataset):
    """
    PyTorch Dataset for the SEN12MS satellite image retrieval benchmark.

    Reads paired SAR / optical / land-cover GeoTIFF triplets referenced
    by a metadata CSV file and returns normalised tensors per sample.

    Parameters
    ----------
    csv_path:
        Path to one of the split CSV files (train.csv / val.csv / test.csv).
        Must contain columns: patch_key, region_id, s1_path, s2_path, lc_path.

    data_root:
        Absolute root of the raw dataset directory (e.g. ``data/raw``).
        The relative paths stored in the CSV are joined to this root.

    normalize:
        Apply the standard S1 / S2 normalisation described in the module
        docstring. Default ``True``. When ``False``, raw numpy arrays are
        converted to float32 tensors without any scaling.

    transform:
        Optional callable applied to the output dict **after** tensor
        conversion. Receives and must return the sample dict. Suitable for
        geometry augmentation or additional channel selection.

    validate_on_init:
        When ``True``, verify that every file in the CSV exists before
        returning. Adds startup overhead but surfaces data issues early.
        Default ``False``.
    """

    def __init__(
        self,
        csv_path:         Path | str,
        data_root:        Path | str,
        normalize:        bool                          = True,
        transform:        Optional[Callable[..., Any]] = None,
        validate_on_init: bool                          = False,
    ) -> None:
        super().__init__()

        self.csv_path  = Path(csv_path).resolve()
        self.data_root = Path(data_root).resolve()
        self.normalize = normalize
        self.transform = transform

        # ── Validate inputs ───────────────────────────────────────────────
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {self.csv_path}\n"
                "Run backend/scripts/build_dataset.py first."
            )
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory not found: {self.data_root}"
            )

        # ── Load metadata ─────────────────────────────────────────────────
        self._df: pd.DataFrame = self._load_csv()

        logger.info(
            "SEN12MSDataset initialised  |  split=%s  |  samples=%d  |  root=%s",
            self.csv_path.stem,
            len(self._df),
            self.data_root,
        )

        # ── Optional early file validation ────────────────────────────────
        if validate_on_init:
            self._validate_all_files()

    # ── CSV loading ───────────────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        """
        Read and validate the metadata CSV.

        Returns
        -------
        pd.DataFrame with required columns confirmed present.

        Raises
        ------
        ValueError: Required columns are missing or patch_key contains duplicates.
        """
        required = {"patch_key", "region_id", "s1_path", "s2_path", "lc_path"}
        df = pd.read_csv(self.csv_path, dtype=str)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {self.csv_path.name} is missing columns: {sorted(missing)}"
            )

        dupes = df.duplicated(subset="patch_key", keep=False)
        if dupes.any():
            n = int(dupes.sum())
            raise ValueError(
                f"{n} duplicate patch_key entries found in {self.csv_path.name}. "
                "Re-run generate_metadata.py to fix."
            )

        # Convert region_id to int for the returned dict
        df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce").fillna(0).astype(int)
        return df.reset_index(drop=True)

    # ── File validation ───────────────────────────────────────────────────

    def _validate_all_files(self) -> None:
        """
        Verify every s1 / s2 / lc path in the CSV resolves to an existing file.

        Logs a warning for every missing file and raises ``FileNotFoundError``
        at the end if any files are absent.

        Raises
        ------
        FileNotFoundError: One or more files are missing.
        """
        logger.info("Validating %d file triplets (this may take a moment)…", len(self._df))
        missing_files: list[str] = []

        for idx, row in self._df.iterrows():
            for col in ("s1_path", "s2_path", "lc_path"):
                full = self.data_root / row[col]
                if not full.exists():
                    missing_files.append(str(full))
                    logger.warning("Missing file [row %d]: %s", idx, full)

        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} files referenced in the CSV do not exist. "
                "First missing: " + missing_files[0]
            )

        logger.info("File validation passed — all triplets present.")

    # ── TIFF loading ──────────────────────────────────────────────────────

    def _read_tiff(
        self,
        path: Path,
        expected_bands: int,
        label: str,
    ) -> np.ndarray:
        """
        Open a GeoTIFF with rasterio and return all bands as a numpy array.

        Parameters
        ----------
        path:           Absolute path to the .tif file.
        expected_bands: Expected number of bands (checked after reading).
        label:          Human-readable label for error messages ("s1", "s2", "lc").

        Returns
        -------
        np.ndarray of shape [bands, H, W] in the file's native dtype.

        Raises
        ------
        FileNotFoundError: File does not exist.
        IOError:           File cannot be opened by rasterio.
        ValueError:        Band count or spatial shape does not match expectations.
        """
        if not path.exists():
            raise FileNotFoundError(f"[{label}] TIFF not found: {path}")

        try:
            with rasterio.open(path) as src:
                data = src.read()   # [bands, H, W]
        except rasterio.errors.RasterioIOError as exc:
            raise IOError(f"[{label}] Cannot open TIFF {path}: {exc}") from exc

        # Band count check
        if data.shape[0] != expected_bands:
            raise ValueError(
                f"[{label}] Expected {expected_bands} bands, "
                f"got {data.shape[0]} in {path.name}"
            )

        # Spatial shape check
        if data.shape[1] != PATCH_H or data.shape[2] != PATCH_W:
            raise ValueError(
                f"[{label}] Expected {PATCH_H}×{PATCH_W}, "
                f"got {data.shape[1]}×{data.shape[2]} in {path.name}"
            )

        return data

    # ── Normalisation ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_s1(arr: np.ndarray) -> np.ndarray:
        """
        Normalise Sentinel-1 SAR from dB scale to [0, 1].

        dB values are clipped to [S1_MIN, S1_MAX] then linearly scaled:
            x_norm = (x - S1_MIN) / (S1_MAX - S1_MIN)

        Parameters
        ----------
        arr: [2, H, W] float32 array in dB scale.

        Returns
        -------
        [2, H, W] float32 array with values in [0, 1].
        """
        arr = arr.astype(np.float32)
        arr = np.clip(arr, S1_MIN, S1_MAX)
        return (arr - S1_MIN) / (S1_MAX - S1_MIN)

    @staticmethod
    def _normalize_s2(arr: np.ndarray) -> np.ndarray:
        """
        Normalise Sentinel-2 optical from raw uint16 to [0, 1] reflectance.

        Raw values represent reflectance × 10 000. Divided by S2_SCALE and
        clipped to [0, 1] to handle any out-of-range sensor readings.

        Parameters
        ----------
        arr: [13, H, W] uint16 array (raw Sentinel-2 L1C / L2A values).

        Returns
        -------
        [13, H, W] float32 array with values in [0, 1].
        """
        return np.clip(arr.astype(np.float32) / S2_SCALE, 0.0, 1.0)

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load and return one sample.

        Parameters
        ----------
        idx: Integer index in [0, len(dataset)).

        Returns
        -------
        dict with keys:
            "patch_key" : str
            "region_id" : int
            "s1"        : Tensor[float32]  shape [2,  256, 256]  normalised to [0, 1]
            "s2"        : Tensor[float32]  shape [13, 256, 256]  normalised to [0, 1]
            "lc"        : Tensor[int64]    shape [4,  256, 256]  raw class labels

        Raises
        ------
        FileNotFoundError: A TIFF file does not exist on disk.
        IOError:           A TIFF file cannot be opened by rasterio.
        ValueError:        A TIFF has unexpected bands or spatial shape.
        """
        row = self._df.iloc[idx]

        s1_path = self.data_root / row["s1_path"]
        s2_path = self.data_root / row["s2_path"]
        lc_path = self.data_root / row["lc_path"]

        # Read raw arrays
        s1_arr = self._read_tiff(s1_path, S1_BANDS, "s1")   # [2,  H, W] float32
        s2_arr = self._read_tiff(s2_path, S2_BANDS, "s2")   # [13, H, W] uint16
        lc_arr = self._read_tiff(lc_path, LC_BANDS, "lc")   # [4,  H, W] uint8

        # Normalise
        if self.normalize:
            s1_arr = self._normalize_s1(s1_arr)   # → float32 [0, 1]
            s2_arr = self._normalize_s2(s2_arr)   # → float32 [0, 1]
        else:
            s1_arr = s1_arr.astype(np.float32)
            s2_arr = s2_arr.astype(np.float32)

        # Convert to tensors
        sample: dict[str, Any] = {
            "patch_key": str(row["patch_key"]),
            "region_id": int(row["region_id"]),
            "s1":        torch.from_numpy(s1_arr),                        # float32
            "s2":        torch.from_numpy(s2_arr),                        # float32
            "lc":        torch.from_numpy(lc_arr.astype(np.int64)),       # int64
        }

        # Optional user-supplied transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def patch_keys(self) -> list[str]:
        """Return the ordered list of patch_key strings in this split."""
        return self._df["patch_key"].tolist()

    @property
    def region_ids(self) -> list[int]:
        """Return the ordered list of region_id integers in this split."""
        return self._df["region_id"].tolist()

    def __repr__(self) -> str:
        return (
            f"SEN12MSDataset("
            f"split={self.csv_path.stem!r}, "
            f"samples={len(self)}, "
            f"normalize={self.normalize})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloader(
    dataset:     SEN12MSDataset,
    batch_size:  int  = 32,
    shuffle:     bool = True,
    num_workers: int  = 0,
    pin_memory:  bool | None = None,
    drop_last:   bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a ``SEN12MSDataset``.

    Parameters
    ----------
    dataset:
        A fully initialised ``SEN12MSDataset`` instance.
    batch_size:
        Number of samples per mini-batch. Default 32.
    shuffle:
        Randomly shuffle the dataset each epoch. Typically ``True`` for
        training and ``False`` for validation / test. Default ``True``.
    num_workers:
        Number of sub-processes for data loading.
        Set ``0`` for the main process only (Windows-friendly default).
        On Linux/macOS, ``4`` or ``8`` is typical for GPU training.
    pin_memory:
        If ``None``, auto-detected: ``True`` when CUDA is available,
        ``False`` otherwise. Pinned memory speeds up host→GPU transfer.
    drop_last:
        Drop the last incomplete batch if dataset size is not divisible
        by batch_size. Default ``False``.
    prefetch_factor:
        Number of batches pre-loaded per worker. Only effective when
        ``num_workers > 0``. Default ``None`` (uses PyTorch default of 2).

    Returns
    -------
    torch.utils.data.DataLoader configured for the given dataset.

    Examples
    --------
    >>> ds = SEN12MSDataset("data/metadata/train.csv", "data/raw")
    >>> loader = create_dataloader(ds, batch_size=8, shuffle=True)
    >>> batch = next(iter(loader))
    >>> batch["s1"].shape   # torch.Size([8, 2, 256, 256])
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs: dict[str, Any] = dict(
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = drop_last,
    )

    # prefetch_factor only valid when num_workers > 0
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)

    logger.info(
        "DataLoader created  |  samples=%d  |  batch_size=%d  |  "
        "batches=%d  |  shuffle=%s  |  workers=%d  |  pin_memory=%s",
        len(dataset),
        batch_size,
        len(loader),
        shuffle,
        num_workers,
        pin_memory,
    )

    return loader
