#!/usr/bin/env python3
"""
GeoSense-X — BEN-14K OneDrive Pre-downloader
=============================================
Triggers on-demand download of all OneDrive RECALL_ON_DATA_ACCESS (.npy) files
in the BEN-14K dataset using a thread pool, so the evaluation script can read
them at full speed afterwards.

Usage:
    python ml/predownload_ben14k.py                  (default 16 workers)
    python ml/predownload_ben14k.py --workers 32
    python ml/predownload_ben14k.py --check-only     (count cloud vs local)
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Locate project root
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

# Auto-relaunch with venv if needed
_VENV_PY = _PROJECT_DIR / ".venv311" / "Scripts" / "python.exe"
if not os.environ.get("_GEOSENSE_VENV_ACTIVE") and _VENV_PY.exists() \
        and Path(sys.executable).resolve() != _VENV_PY.resolve():
    import subprocess
    env = os.environ.copy()
    env["_GEOSENSE_VENV_ACTIVE"] = "1"
    sys.exit(subprocess.run([str(_VENV_PY)] + sys.argv, env=env).returncode)

import numpy as np

# Windows OneDrive file attribute constants
_GA = ctypes.windll.kernel32.GetFileAttributesW
_FILE_ATTR_OFFLINE = 0x00001000
_FILE_ATTR_RECALL  = 0x00400000


def _attrs(p: Path) -> int:
    return _GA(str(p))


def _is_recall(p: Path) -> bool:
    a = _attrs(p)
    return a != 0xFFFFFFFF and bool(a & _FILE_ATTR_RECALL)


def _is_local(p: Path) -> bool:
    a = _attrs(p)
    return a != 0xFFFFFFFF and not (a & (_FILE_ATTR_OFFLINE | _FILE_ATTR_RECALL))


def download_file(path: Path, max_retries: int = 4) -> tuple[Path, bool, float]:
    """Read a file to trigger OneDrive download.  Returns (path, success, seconds)."""
    t0 = time.perf_counter()
    if _is_local(path):
        return path, True, 0.0   # already local
    for attempt in range(max_retries):
        try:
            np.load(str(path))
            return path, True, time.perf_counter() - t0
        except OSError as exc:
            if exc.errno == 22 and attempt < max_retries - 1:
                time.sleep(10 * (2 ** attempt))
                continue
            return path, False, time.perf_counter() - t0
    return path, False, time.perf_counter() - t0


def collect_cloud_files(root: Path) -> list[Path]:
    """Return all .npy files that need downloading."""
    return [p for p in root.rglob("*.npy") if _is_recall(p)]


def check_status(root: Path) -> None:
    files = list(root.rglob("*.npy"))
    local  = sum(1 for p in files if _is_local(p))
    recall = sum(1 for p in files if _is_recall(p))
    print(f"Total .npy files : {len(files)}")
    print(f"Local (ready)    : {local}  ({100*local/max(len(files),1):.1f}%)")
    print(f"RECALL (pending) : {recall}  ({100*recall/max(len(files),1):.1f}%)")
    print(f"Other (offline)  : {len(files)-local-recall}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-download BEN-14K OneDrive files")
    ap.add_argument("--workers", type=int, default=16, help="parallel download threads")
    ap.add_argument("--root", default=str(_PROJECT_DIR / "data" / "raw" / "benv1_14k"),
                    help="dataset root directory")
    ap.add_argument("--check-only", action="store_true", help="just print counts, no download")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset root: {root}")
    check_status(root)

    if args.check_only:
        return

    cloud_files = collect_cloud_files(root)
    total = len(cloud_files)
    if total == 0:
        print("All files already local — nothing to download.")
        return

    print(f"\nDownloading {total} cloud files with {args.workers} workers ...")
    print("This may take several hours.  Press Ctrl+C to stop early.\n")

    t_start = time.perf_counter()
    done = ok = failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, p): p for p in cloud_files}
        try:
            for fut in as_completed(futures):
                path, success, secs = fut.result()
                done += 1
                if success:
                    ok += 1
                else:
                    failed += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.perf_counter() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta  = (total - done) / rate if rate > 0 else float("inf")
                    print(
                        f"  {done}/{total}  ok={ok}  failed={failed}  "
                        f"rate={rate:.2f} files/s  ETA={eta/60:.0f} min",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nInterrupted — downloaded so far:")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed/60:.1f} min  —  ok={ok}  failed={failed}")
    print("\nFinal status:")
    check_status(root)


if __name__ == "__main__":
    main()
