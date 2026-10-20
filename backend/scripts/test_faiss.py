#!/usr/bin/env python3
"""
GeoSense-X — FAISS Index Test Script
=======================================
Loads the saved FAISS index and the companion patch-key CSV, uses
embedding #0 as a query vector, retrieves the top-5 nearest neighbours,
and prints a full diagnostic report.

Checks performed
─────────────────
  1.  Index file exists and is readable.
  2.  ntotal > 0  (index contains embeddings).
  3.  Dimension == 512.
  4.  Query embedding is L2-normalised (norm ≈ 1.0).
  5.  Top-1 hit is the query itself  (index 0, score ≈ 1.0).
  6.  All similarity scores are in [−1, 1]  (valid cosine range).
  7.  Scores are sorted descending.
  8.  Patch-key CSV aligns with index  (same row count).

Usage
─────
  python backend/scripts/test_faiss.py

  # Test a different index or query index
  python backend/scripts/test_faiss.py --index data/index/faiss.index --query-idx 5

  # Top-10 instead of top-5
  python backend/scripts/test_faiss.py --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = Path(__file__).resolve().parent      # backend/scripts/
_PROJECT_DIR = _SCRIPT_DIR.parent.parent            # ps11/

# ─────────────────────────────────────────────────────────────────────────────
# Expected constants
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_DIM:     int   = 512
NORM_TOL:         float = 1e-4
SCORE_TOP1_TOL:   float = 1e-4     # top-1 self-score must be ≥ 1 − this
SEP  = "=" * 60
SEP2 = "-" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Check helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(label: str, detail: str = "") -> bool:
    tail = f"  ({detail})" if detail else ""
    print(f"    [PASS]  {label}{tail}")
    return True


def _fail(label: str, detail: str = "") -> bool:
    tail = f"  ({detail})" if detail else ""
    print(f"    [FAIL]  {label}{tail}")
    return False


def _check(cond: bool, label: str, detail: str = "") -> bool:
    return _ok(label, detail) if cond else _fail(label, detail)


# ─────────────────────────────────────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────────────────────────────────────

def test_faiss(
    index_path:  Path,
    keys_path:   Path,
    emb_path:    Path,
    top_k:       int = 5,
    query_idx:   int = 0,
) -> int:
    """
    Full diagnostic test of the FAISS index.

    Parameters
    ----------
    index_path : Path to the saved FAISS index (``faiss.index``).
    keys_path  : Path to the patch-key CSV  (``train_patch_keys.csv``).
    emb_path   : Path to the source embeddings (``train_embeddings.npy``).
    top_k      : Number of nearest neighbours to retrieve.
    query_idx  : Row index used as the query embedding.

    Returns
    -------
    int  0 = all checks passed,  1 = one or more failures.
    """
    results: list[bool] = []

    print()
    print(SEP)
    print("  GeoSense-X  -  FAISS Index Test")
    print(SEP)
    print(f"  Index      :  {index_path}")
    print(f"  Keys CSV   :  {keys_path}")
    print(f"  Embeddings :  {emb_path}")
    print(f"  Top-K      :  {top_k}")
    print(f"  Query idx  :  {query_idx}")
    print(SEP)

    # ── Check 1: index file exists ────────────────────────────────────────
    print("\n[1] File existence")
    ok = _check(index_path.exists(), "faiss.index exists")
    results.append(ok)
    if not ok:
        print()
        print(f"  ERROR: {index_path} not found.")
        print("  Run:  python backend/ml/faiss_index.py")
        print()
        return 1

    # ── Load index ────────────────────────────────────────────────────────
    print("\n[2] Loading FAISS index ...")
    try:
        index = faiss.read_index(str(index_path))
        size_kb = index_path.stat().st_size / 1024
        print(f"    Loaded  {index_path.name}  "
              f"({index.ntotal} vectors, d={index.d}, {size_kb:.1f} KB)")
    except Exception as exc:
        _fail("Read index", str(exc))
        return 1

    # ── Check 2: non-empty ────────────────────────────────────────────────
    print("\n[3] Index integrity")
    results.append(_check(index.ntotal > 0, "Index is non-empty",
                          f"ntotal={index.ntotal}"))
    results.append(_check(index.d == EXPECTED_DIM, f"Dimension == {EXPECTED_DIM}",
                          f"got {index.d}"))
    results.append(_check(
        isinstance(index, faiss.IndexFlatIP),
        "Index type is IndexFlatIP",
        f"{index.__class__.__name__}",
    ))

    # ── Load embeddings for query ─────────────────────────────────────────
    print("\n[4] Loading embeddings for query ...")
    if not emb_path.exists():
        _fail("Embeddings file exists", str(emb_path))
        results.append(False)
        # Fall back: reconstruct query from the index using reconstruct()
        try:
            query = np.zeros((1, index.d), dtype=np.float32)
            index.reconstruct(query_idx, query[0])
            print(f"    Reconstructed embedding #{query_idx} from index.")
        except Exception:
            print("    Cannot reconstruct query — aborting search test.")
            return 1
    else:
        results.append(True)
        embeddings = np.load(str(emb_path)).astype(np.float32)

        if query_idx >= len(embeddings):
            logger_msg = (
                f"query_idx={query_idx} >= N={len(embeddings)}; "
                f"falling back to idx 0"
            )
            print(f"    WARNING: {logger_msg}")
            query_idx = 0

        query = embeddings[query_idx: query_idx + 1].copy()  # [1, D]

    # ── Check 3: query norm ───────────────────────────────────────────────
    print("\n[5] Query embedding properties")
    q_norm = float(np.linalg.norm(query))
    results.append(_check(
        abs(q_norm - 1.0) < NORM_TOL,
        "Query is L2-normalised",
        f"norm={q_norm:.6f}",
    ))

    # ── FAISS search ──────────────────────────────────────────────────────
    print(f"\n[6] Searching: top-{top_k} neighbours for embedding #{query_idx} ...")
    faiss.normalize_L2(query)                          # defensive re-normalise
    D, I = index.search(query, top_k)                 # D,I shape [1, top_k]
    scores  = D[0]                                     # [top_k]
    indices = I[0]                                     # [top_k]

    print(f"\n    {'Rank':<6}  {'FAISS idx':>10}  {'Score (cosine)':>16}  Patch Key")
    print(f"    {'----':<6}  {'---------':>10}  {'--------------':>16}  ---------")

    # Load patch keys for display
    keys_df: pd.DataFrame | None = None
    if keys_path.exists():
        try:
            keys_df = pd.read_csv(keys_path, dtype=str)
        except Exception:
            pass

    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        patch_key = "—"
        if keys_df is not None and idx < len(keys_df):
            patch_key = keys_df.iloc[int(idx)]["patch_key"]
        self_marker = "  <-- query" if int(idx) == query_idx else ""
        print(f"    {rank:<6}  {int(idx):>10}  {float(score):>16.8f}  "
              f"{patch_key}{self_marker}")

    # ── Check 4: top-1 is the query ───────────────────────────────────────
    print(f"\n[7] Search result checks")
    top1_is_self = (int(indices[0]) == query_idx)
    results.append(_check(
        top1_is_self,
        f"Top-1 hit is the query (idx {query_idx})",
        f"got idx {int(indices[0])}",
    ))
    results.append(_check(
        float(scores[0]) >= 1.0 - SCORE_TOP1_TOL,
        f"Top-1 score >= {1.0 - SCORE_TOP1_TOL:.4f}  (self-similarity)",
        f"got {float(scores[0]):.8f}",
    ))

    # ── Check 5: all scores in valid cosine range ─────────────────────────
    all_in_range = bool((scores >= -1.0 - 1e-5).all() and (scores <= 1.0 + 1e-5).all())
    results.append(_check(
        all_in_range,
        "All scores in [-1, 1]  (valid cosine range)",
        f"min={scores.min():.6f}  max={scores.max():.6f}",
    ))

    # ── Check 6: descending order ─────────────────────────────────────────
    is_sorted = bool((scores[:-1] >= scores[1:]).all())
    results.append(_check(
        is_sorted,
        "Scores are sorted descending",
        f"scores={scores.round(4).tolist()}",
    ))

    # ── Check 7: no invalid indices ───────────────────────────────────────
    all_valid_idx = bool((indices >= 0).all() and (indices < index.ntotal).all())
    results.append(_check(
        all_valid_idx,
        "All returned indices are valid",
        f"range [0, {index.ntotal - 1}]",
    ))

    # ── Check 8: patch-key CSV alignment ─────────────────────────────────
    print("\n[8] Patch-key CSV alignment")
    if keys_path.exists() and keys_df is not None:
        csv_n = len(keys_df)
        results.append(_check(
            csv_n == index.ntotal,
            "CSV row count == index.ntotal",
            f"csv={csv_n}  index={index.ntotal}",
        ))
        results.append(_check(
            "patch_key" in keys_df.columns,
            "Column 'patch_key' present",
        ))
    else:
        print(f"    [SKIP]  CSV not found at {keys_path}")

    # ── Optional: load index_meta.json ────────────────────────────────────
    meta_path = index_path.parent / "index_meta.json"
    if meta_path.exists():
        print(f"\n[9] Index metadata  ({meta_path.name})")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        for k, v in meta.items():
            print(f"    {k:<18}:  {v}")

    # ── Summary ───────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass

    print()
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  Index         :  {index_path.name}")
    print(f"  ntotal        :  {index.ntotal:,}")
    print(f"  Dimension     :  {index.d}  "
          f"{'[OK]' if index.d == EXPECTED_DIM else '[EXPECTED ' + str(EXPECTED_DIM) + ']'}")
    print(f"  Index type    :  {index.__class__.__name__}")
    print(f"  Index size    :  {index_path.stat().st_size / 1024:.2f} KB")
    print()
    print(f"  Query index   :  #{query_idx}")
    print(f"  Top-{top_k} results  :  {list(indices)}")
    print(f"  Top-{top_k} scores   :  {[round(float(s), 6) for s in scores]}")
    print()
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
    default_index = _PROJECT_DIR / "data" / "index" / "faiss.index"
    default_keys  = _PROJECT_DIR / "data" / "embeddings" / "train_patch_keys.csv"
    default_emb   = _PROJECT_DIR / "data" / "embeddings" / "train_embeddings.npy"

    p = argparse.ArgumentParser(
        prog="test_faiss",
        description="Test the FAISS index with a query embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python backend/scripts/test_faiss.py
  python backend/scripts/test_faiss.py --top-k 10
  python backend/scripts/test_faiss.py --query-idx 42
""",
    )
    p.add_argument("--index",     type=Path, default=default_index,
                   help="Path to faiss.index")
    p.add_argument("--keys",      type=Path, default=default_keys,
                   help="Path to patch_keys.csv")
    p.add_argument("--embeddings",type=Path, default=default_emb,
                   help="Path to embeddings .npy (for query reconstruction)")
    p.add_argument("--top-k",     type=int,  default=5,
                   help="Number of neighbours to retrieve (default: 5)")
    p.add_argument("--query-idx", type=int,  default=0,
                   help="Row index of the query embedding (default: 0)")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return test_faiss(
        index_path = args.index,
        keys_path  = args.keys,
        emb_path   = args.embeddings,
        top_k      = args.top_k,
        query_idx  = args.query_idx,
    )


if __name__ == "__main__":
    sys.exit(main())
