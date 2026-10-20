#!/usr/bin/env python3
"""
GeoSense-X — Embedding Verification Script
============================================
Loads the generated embeddings and patch-key CSV, runs a suite of
validation checks, and prints a concise diagnostic report.

Checks performed
─────────────────
  1. Files exist and are readable.
  2. Shape is [N, 512]  (N = number of samples, 512 = embed_dim).
  3. Embedding dimension == 512.
  4. dtype is float32.
  5. No NaN or Inf values.
  6. All L2 norms == 1.0  (±1e-4)  — confirming unit hypersphere.
  7. patch_keys CSV row count matches embedding row count.
  8. No duplicate patch_key entries.
  9. Cosine self-similarity of any embedding with itself == 1.0.
 10. First and last embedding printed for a visual sanity check.

Usage
─────
  python backend/scripts/test_embeddings.py

  # Test a different split
  python backend/scripts/test_embeddings.py --split val

  # Custom embedding directory
  python backend/scripts/test_embeddings.py \\
      --embeddings-dir data/embeddings \\
      --split train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Project root detection
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = Path(__file__).resolve().parent          # backend/scripts/
_BACKEND_DIR = _SCRIPT_DIR.parent                       # backend/
_PROJECT_DIR = _BACKEND_DIR.parent                      # ps11/

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_EMBED_DIM: int   = 512
NORM_TOLERANCE:     float = 1e-4
SEP  = "=" * 58
SEP2 = "-" * 58


# ─────────────────────────────────────────────────────────────────────────────
# Individual check helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(label: str, detail: str = "") -> bool:
    suffix = f"  ({detail})" if detail else ""
    print(f"    [PASS]  {label}{suffix}")
    return True


def _fail(label: str, detail: str = "") -> bool:
    suffix = f"  ({detail})" if detail else ""
    print(f"    [FAIL]  {label}{suffix}")
    return False


def _check(cond: bool, label: str, detail: str = "") -> bool:
    return _pass(label, detail) if cond else _fail(label, detail)


# ─────────────────────────────────────────────────────────────────────────────
# Main verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_embeddings(
    split:          str  = "train",
    embeddings_dir: Path = _PROJECT_DIR / "data" / "embeddings",
) -> int:
    """
    Run all embedding checks and print a report.

    Parameters
    ----------
    split          : "train" | "val" | "test"
    embeddings_dir : directory that contains *_embeddings.npy and *_patch_keys.csv

    Returns
    -------
    int  0 if all checks pass, 1 otherwise.
    """
    emb_path  = embeddings_dir / f"{split}_embeddings.npy"
    keys_path = embeddings_dir / f"{split}_patch_keys.csv"

    print()
    print(SEP)
    print("  GeoSense-X  -  Embedding Verification")
    print(SEP)
    print(f"  Split          :  {split}")
    print(f"  Embeddings     :  {emb_path}")
    print(f"  Patch keys     :  {keys_path}")
    print(SEP)

    results: list[bool] = []

    # ── Check 1: files exist ─────────────────────────────────────────────
    print("\n[1] File existence")
    emb_ok  = _check(emb_path.exists(),  "embeddings .npy exists", str(emb_path))
    keys_ok = _check(keys_path.exists(), "patch_keys .csv exists", str(keys_path))
    results += [emb_ok, keys_ok]

    if not emb_ok:
        print()
        print("  ERROR: embeddings file not found.")
        print(f"  Run:  python backend/ml/generate_embeddings.py --split {split}")
        print()
        return 1

    # ── Load embeddings ───────────────────────────────────────────────────
    print("\n[2] Loading embeddings ...")
    try:
        embeddings = np.load(str(emb_path))
        print(f"    Loaded  {emb_path.name}  ->  {embeddings.shape}  {embeddings.dtype}")
    except Exception as exc:
        _fail("Load embeddings", str(exc))
        return 1

    n, d = embeddings.shape

    # ── Check 2: shape ────────────────────────────────────────────────────
    print("\n[3] Shape checks")
    results.append(_check(embeddings.ndim == 2, "Array is 2-dimensional",
                          f"ndim={embeddings.ndim}"))
    results.append(_check(d == EXPECTED_EMBED_DIM, f"Embedding dim == {EXPECTED_EMBED_DIM}",
                          f"got {d}"))
    results.append(_check(n > 0, "At least one embedding present", f"N={n}"))

    # ── Check 3: dtype ────────────────────────────────────────────────────
    print("\n[4] Data type")
    results.append(_check(embeddings.dtype == np.float32, "dtype == float32",
                          f"got {embeddings.dtype}"))

    # ── Check 4: NaN / Inf ────────────────────────────────────────────────
    print("\n[5] NaN / Inf values")
    n_nan = int(np.isnan(embeddings).sum())
    n_inf = int(np.isinf(embeddings).sum())
    results.append(_check(n_nan == 0, "No NaN values",  f"{n_nan} NaNs"))
    results.append(_check(n_inf == 0, "No Inf values",  f"{n_inf} Infs"))

    # ── Check 5: L2 norms ────────────────────────────────────────────────
    print("\n[6] L2 norms (should all be 1.0)")
    norms      = np.linalg.norm(embeddings, axis=1)   # [N]
    norms_ok   = np.allclose(norms, 1.0, atol=NORM_TOLERANCE)
    n_bad      = int(~np.isclose(norms, 1.0, atol=NORM_TOLERANCE).sum())
    results.append(_check(
        norms_ok,
        f"All norms within +/-{NORM_TOLERANCE} of 1.0",
        f"mean={norms.mean():.6f}  std={norms.std():.6f}  bad={n_bad}",
    ))
    print(f"    min={norms.min():.6f}  max={norms.max():.6f}  "
          f"mean={norms.mean():.6f}  std={norms.std():.6f}")

    # ── Check 6: patch keys CSV ───────────────────────────────────────────
    print("\n[7] Patch keys CSV")
    if keys_ok:
        try:
            keys_df = pd.read_csv(keys_path, dtype=str)
            results.append(_check(
                len(keys_df) == n,
                "Row count matches embeddings",
                f"csv_rows={len(keys_df)}  emb_rows={n}",
            ))
            results.append(_check(
                "patch_key" in keys_df.columns,
                "Column 'patch_key' present",
            ))
            results.append(_check(
                "region_id" in keys_df.columns,
                "Column 'region_id' present",
            ))
            n_dupes = int(keys_df.duplicated(subset="patch_key").sum())
            results.append(_check(
                n_dupes == 0,
                "No duplicate patch_key entries",
                f"{n_dupes} duplicates",
            ))
        except Exception as exc:
            _fail("Load patch_keys CSV", str(exc))
            keys_df = pd.DataFrame()

    # ── Check 7: cosine self-similarity sanity ────────────────────────────
    print("\n[8] Cosine self-similarity (e[0] . e[0] should be 1.0)")
    if n > 0:
        cos_self = float(embeddings[0] @ embeddings[0])
        results.append(_check(
            abs(cos_self - 1.0) < 1e-4,
            "Cosine self-sim == 1.0",
            f"got {cos_self:.8f}",
        ))

    # ── Print first and last embedding ───────────────────────────────────
    print()
    print(SEP2)
    print(f"  First embedding  (idx 0)  — first 10 values:")
    print(f"    {embeddings[0, :10]}")
    if n > 1:
        print(f"\n  Last embedding   (idx {n-1})  — first 10 values:")
        print(f"    {embeddings[n-1, :10]}")

    # ── Pairwise cosine similarity sample ─────────────────────────────────
    if n >= 2:
        n_sample  = min(500, n)
        idx       = np.random.default_rng(0).choice(n, size=n_sample, replace=False)
        sub       = embeddings[idx]                   # [n_sample, d]
        cos_mat   = sub @ sub.T                       # [n_sample, n_sample]
        np.fill_diagonal(cos_mat, np.nan)
        pairwise  = cos_mat[~np.isnan(cos_mat)]
        print()
        print(f"  Pairwise cosine similarity (random {n_sample} samples, off-diagonal):")
        print(f"    mean={pairwise.mean():.4f}  "
              f"std={pairwise.std():.4f}  "
              f"min={pairwise.min():.4f}  "
              f"max={pairwise.max():.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  Embeddings file  :  {emb_path.name}")
    print(f"  Shape            :  ({n:,}, {d})")
    print(f"  Embedding dim    :  {d}  {'[OK]' if d == EXPECTED_EMBED_DIM else '[EXPECTED ' + str(EXPECTED_EMBED_DIM) + ']'}")
    print(f"  dtype            :  {embeddings.dtype}")
    print(f"  NaN values       :  {n_nan}")
    print(f"  Inf values       :  {n_inf}")
    print(f"  L2 norm (mean)   :  {norms.mean():.6f}")
    print(f"  File size        :  {emb_path.stat().st_size / 1_048_576:.2f} MB")
    print()

    n_pass = sum(results)
    n_fail = len(results) - n_pass

    if n_fail == 0:
        print(f"  Result  :  ALL {n_pass} CHECKS PASSED  [OK]")
    else:
        print(f"  Result  :  {n_pass} passed  |  {n_fail} FAILED  [ERROR]")

    print(SEP)
    print()

    return 0 if n_fail == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_embeddings",
        description="Verify generated DINOv2 embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python backend/scripts/test_embeddings.py
  python backend/scripts/test_embeddings.py --split val
  python backend/scripts/test_embeddings.py --split test
""",
    )
    p.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test"],
        help="Split to verify (default: train)",
    )
    p.add_argument(
        "--embeddings-dir", type=Path,
        default=_PROJECT_DIR / "data" / "embeddings",
        help="Directory containing *_embeddings.npy and *_patch_keys.csv",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return verify_embeddings(
        split          = args.split,
        embeddings_dir = args.embeddings_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
