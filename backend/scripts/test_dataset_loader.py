"""
GeoSense-X — Dataset Loader Test Script
=========================================
Loads train.csv, creates a DataLoader, iterates over the first batch,
and prints a comprehensive shape / value report to verify the loader works
correctly end-to-end.

Usage
─────
  python backend/scripts/test_dataset_loader.py

  # Faster smoke test with custom batch size
  python backend/scripts/test_dataset_loader.py --batch-size 4 --num-workers 0

  # Test validation split
  python backend/scripts/test_dataset_loader.py --split val
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# ── resolve project root so imports work from any cwd ─────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # ps11/
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from ml.dataset_loader import (
    S1_BANDS, S1_MAX, S1_MIN,
    S2_BANDS, S2_SCALE,
    LC_BANDS, PATCH_H, PATCH_W,
    SEN12MSDataset,
    create_dataloader,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SEP  = "=" * 60
SEP2 = "-" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_shape(t: torch.Tensor) -> str:
    """Return a compact shape / dtype string."""
    return f"[{', '.join(str(d) for d in t.shape)}]  dtype={t.dtype}"


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    value_min: float,
    value_max: float,
) -> bool:
    """
    Validate a single tensor's shape, dtype, and value range.

    Prints PASS / FAIL for each check and returns True iff all checks pass.
    """
    all_ok = True

    # Shape
    actual_shape = tuple(tensor.shape)
    shape_ok = actual_shape == expected_shape
    print(f"  Shape    :  {actual_shape}  {'[PASS]' if shape_ok else '[FAIL] expected ' + str(expected_shape)}")
    all_ok = all_ok and shape_ok

    # Dtype
    dtype_ok = tensor.dtype == expected_dtype
    print(f"  Dtype    :  {tensor.dtype}  {'[PASS]' if dtype_ok else '[FAIL] expected ' + str(expected_dtype)}")
    all_ok = all_ok and dtype_ok

    # Value range
    actual_min = float(tensor.min())
    actual_max = float(tensor.max())
    range_ok   = (actual_min >= value_min - 1e-3) and (actual_max <= value_max + 1e-3)
    print(f"  Min      :  {actual_min:>10.4f}  {'[PASS]' if actual_min >= value_min - 1e-3 else '[FAIL] below ' + str(value_min)}")
    print(f"  Max      :  {actual_max:>10.4f}  {'[PASS]' if actual_max <= value_max + 1e-3 else '[FAIL] above ' + str(value_max)}")
    all_ok = all_ok and range_ok

    # NaN / Inf check
    has_nan = bool(torch.isnan(tensor).any())
    has_inf = bool(torch.isinf(tensor).any())
    nan_ok  = not has_nan and not has_inf
    print(f"  NaN/Inf  :  nan={has_nan}  inf={has_inf}  {'[PASS]' if nan_ok else '[FAIL]'}")
    all_ok = all_ok and nan_ok

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    split:       str = "train",
    batch_size:  int = 8,
    num_workers: int = 0,
) -> int:
    """
    Execute the dataset loader test.

    Parameters
    ----------
    split:       Which split CSV to load ("train" / "val" / "test").
    batch_size:  Batch size for the DataLoader.
    num_workers: DataLoader worker count.

    Returns
    -------
    int
        0 if all checks pass, 1 otherwise.
    """
    project_root = _PROJECT_ROOT
    csv_path     = project_root / "data" / "metadata" / f"{split}.csv"
    data_root    = project_root / "data" / "raw"

    print()
    print(SEP)
    print("  GeoSense-X  -  Dataset Loader Test")
    print(SEP)
    print(f"  Split      :  {split}")
    print(f"  CSV        :  {csv_path}")
    print(f"  Data root  :  {data_root}")
    print(f"  Batch size :  {batch_size}")
    print(f"  Workers    :  {num_workers}")
    print(f"  CUDA       :  {torch.cuda.is_available()}")
    print(SEP)

    # ── 1. Create dataset ──────────────────────────────────────────────────
    print("\n[1] Creating SEN12MSDataset …")
    t0 = time.perf_counter()
    try:
        dataset = SEN12MSDataset(
            csv_path         = csv_path,
            data_root        = data_root,
            normalize        = True,
            validate_on_init = False,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset creation failed: %s", exc)
        return 1

    print(f"    Dataset   :  {dataset}")
    print(f"    Size      :  {len(dataset):,} samples  ({time.perf_counter()-t0:.2f}s)")

    # ── 2. Create dataloader ───────────────────────────────────────────────
    print(f"\n[2] Creating DataLoader (batch_size={batch_size}) …")
    loader = create_dataloader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
    )
    print(f"    Batches per epoch  :  {len(loader):,}")
    print(f"    Samples per epoch  :  {len(loader) * batch_size:,}  "
          f"(may differ by up to 1 batch if drop_last=False)")

    # ── 3. Load first batch ────────────────────────────────────────────────
    print(f"\n[3] Loading first batch …")
    t_batch = time.perf_counter()
    try:
        batch = next(iter(loader))
    except StopIteration:
        logger.error("DataLoader is empty — no batches available.")
        return 1
    except Exception as exc:
        logger.exception("Error loading first batch: %s", exc)
        return 1

    elapsed_ms = (time.perf_counter() - t_batch) * 1000
    print(f"    First batch loaded in  {elapsed_ms:.0f} ms")

    # ── 4. Inspect batch keys ──────────────────────────────────────────────
    print(f"\n[4] Batch contents:")
    print(f"    Keys         :  {list(batch.keys())}")
    print(f"    patch_key[0] :  {batch['patch_key'][0]}")
    print(f"    region_id[0] :  {batch['region_id'][0]}")

    # ── 5. Shape + value checks ────────────────────────────────────────────
    print(f"\n[5] Tensor validation:")
    all_pass = True

    # S1
    print(f"\n  {SEP2}")
    print(f"  S1  (Sentinel-1 SAR)   expected [{batch_size}, {S1_BANDS}, {PATCH_H}, {PATCH_W}]  float32  [0,1]")
    print(f"  {SEP2}")
    s1_ok = _check_tensor(
        batch["s1"],
        name           = "s1",
        expected_shape = (batch_size, S1_BANDS, PATCH_H, PATCH_W),
        expected_dtype = torch.float32,
        value_min      = 0.0,
        value_max      = 1.0,
    )
    all_pass = all_pass and s1_ok

    # S2
    print(f"\n  {SEP2}")
    print(f"  S2  (Sentinel-2 Optical) expected [{batch_size}, {S2_BANDS}, {PATCH_H}, {PATCH_W}]  float32  [0,1]")
    print(f"  {SEP2}")
    s2_ok = _check_tensor(
        batch["s2"],
        name           = "s2",
        expected_shape = (batch_size, S2_BANDS, PATCH_H, PATCH_W),
        expected_dtype = torch.float32,
        value_min      = 0.0,
        value_max      = 1.0,
    )
    all_pass = all_pass and s2_ok

    # LC
    print(f"\n  {SEP2}")
    print(f"  LC  (Land Cover Labels) expected [{batch_size}, {LC_BANDS}, {PATCH_H}, {PATCH_W}]  int64  [0,255]")
    print(f"  {SEP2}")
    lc_ok = _check_tensor(
        batch["lc"],
        name           = "lc",
        expected_shape = (batch_size, LC_BANDS, PATCH_H, PATCH_W),
        expected_dtype = torch.int64,
        value_min      = 0.0,
        value_max      = 255.0,
    )
    all_pass = all_pass and lc_ok

    # ── 6. Per-channel statistics ──────────────────────────────────────────
    print(f"\n[6] S1 per-channel stats (after normalisation to [0,1]):")
    print(f"    {'Band':<8}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"    {'----':<8}  {'----':>8}  {'---':>8}  {'---':>8}  {'---':>8}")
    s1 = batch["s1"]
    names_s1 = ["VV", "VH"]
    for b in range(S1_BANDS):
        ch = s1[:, b, :, :]
        print(f"    {names_s1[b]:<8}  {ch.mean():>8.4f}  {ch.std():>8.4f}  "
              f"{ch.min():>8.4f}  {ch.max():>8.4f}")

    print(f"\n[7] S2 per-channel stats (after /10000 normalisation):")
    print(f"    {'Band':<8}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"    {'----':<8}  {'----':>8}  {'---':>8}  {'---':>8}  {'---':>8}")
    s2 = batch["s2"]
    s2_names = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12","B10"]
    for b in range(S2_BANDS):
        ch = s2[:, b, :, :]
        print(f"    {s2_names[b]:<8}  {ch.mean():>8.4f}  {ch.std():>8.4f}  "
              f"{ch.min():>8.4f}  {ch.max():>8.4f}")

    print(f"\n[8] LC class distribution (Band 0 = IGBP primary labels):")
    lc_b0      = batch["lc"][:, 0, :, :]    # [B, H, W] — IGBP
    unique_cls = lc_b0.unique().tolist()
    total_px   = lc_b0.numel()
    print(f"    Unique classes in this batch : {[int(c) for c in unique_cls]}")
    print(f"    {'Class':>8}  {'Pixels':>10}  {'%':>7}")
    print(f"    {'-----':>8}  {'------':>10}  {'---':>7}")
    for cls in sorted(unique_cls):
        count = int((lc_b0 == cls).sum())
        pct   = count / total_px * 100
        print(f"    {int(cls):>8}  {count:>10,}  {pct:>7.2f}%")

    # ── 9. Iteration speed benchmark ──────────────────────────────────────
    N_BATCHES = min(10, len(loader))
    print(f"\n[9] Speed benchmark — {N_BATCHES} batches …")
    t_speed = time.perf_counter()
    loader_iter = iter(loader)
    for _ in range(N_BATCHES):
        _ = next(loader_iter)
    elapsed  = time.perf_counter() - t_speed
    n_images = N_BATCHES * batch_size
    print(f"    {N_BATCHES} batches ({n_images} images) loaded in {elapsed:.2f}s "
          f"({n_images / elapsed:.1f} images/s)")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  Dataset size     :  {len(dataset):>8,} samples")
    print(f"  Batches          :  {len(loader):>8,}")
    print(f"  S1 shape         :  [{batch_size}, {S1_BANDS}, {PATCH_H}, {PATCH_W}]")
    print(f"  S2 shape         :  [{batch_size}, {S2_BANDS}, {PATCH_H}, {PATCH_W}]")
    print(f"  LC shape         :  [{batch_size}, {LC_BANDS}, {PATCH_H}, {PATCH_W}]")
    print(f"  S1 normalised    :  [0, 1]  (clipped dB from [{S1_MIN}, {S1_MAX}])")
    print(f"  S2 normalised    :  [0, 1]  (divided by {int(S2_SCALE)})")
    print(f"  LC               :  int64 class labels  (band 0 = IGBP)")
    print()
    if all_pass:
        print("  Result  :  ALL CHECKS PASSED  [OK]")
    else:
        print("  Result  :  SOME CHECKS FAILED  -- see output above")
    print(SEP)
    print()

    return 0 if all_pass else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_dataset_loader",
        description="Verify SEN12MSDataset end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python backend/scripts/test_dataset_loader.py
  python backend/scripts/test_dataset_loader.py --split val --batch-size 4
  python backend/scripts/test_dataset_loader.py --split test --num-workers 2
""",
    )
    p.add_argument("--split",       type=str, default="train",
                   choices=["train", "val", "test"],
                   help="Which split CSV to load (default: train)")
    p.add_argument("--batch-size",  type=int, default=8,
                   help="DataLoader batch size (default: 8)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default: 0, safe on all OS)")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return run_test(
        split       = args.split,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )


if __name__ == "__main__":
    sys.exit(main())
