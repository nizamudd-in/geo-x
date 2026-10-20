"""
SEN12MS Dataset Loader
======================
Discovers, validates, and loads paired Sentinel-1/Sentinel-2 scene patches
from the SEN12MS dataset and prepares them for DINOv2 embedding generation.

SEN12MS Directory Layout
─────────────────────────
  {root}/
  ├── ROIs1158_spring/
  │   ├── s1_spring/          ← Sentinel-1 SAR  (2 bands: VV, VH — linear scale)
  │   │   └── ROIs1158_spring_s1_p0001.tif
  │   └── s2_spring/          ← Sentinel-2 MSI  (13 bands — uint16 × 10 000)
  │       └── ROIs1158_spring_s2_p0001.tif
  ├── ROIs1868_summer/
  ├── ROIs1970_fall/
  └── ROIs2017_winter/

Sentinel-2 Band Ordering (SEN12MS)
────────────────────────────────────
  Idx  Band   Wavelength  Resolution
   0   B1     443 nm      60 m
   1   B2     490 nm      10 m  ← Blue
   2   B3     560 nm      10 m  ← Green
   3   B4     665 nm      10 m  ← Red
   4   B5     705 nm      20 m
   5   B6     740 nm      20 m
   6   B7     783 nm      20 m
   7   B8     842 nm      10 m  ← NIR
   8   B8A    865 nm      20 m
   9   B9     945 nm      60 m
  10   B11   1610 nm      20 m  ← SWIR-1
  11   B12   2190 nm      20 m  ← SWIR-2
  12   B10   1375 nm      60 m

Reference
─────────
  Schmitt et al., "SEN12MS — A Curated Dataset of Georeferenced Multi-Spectral
  Sentinel-1/2 Imagery for Deep Learning and Data Fusion."
  IGARSS 2019. DOI: 10.14459/2019mp1554803
  Dataset: https://mediatum.ub.tum.de/1554803

Usage
─────
  # Discover scenes
  from dataset_loader import SEN12MSLoader
  loader = SEN12MSLoader("/data/sen12ms", seasons=["spring"], max_scenes=500)
  scenes = loader.discover()

  # Load a single scene
  optical = loader.load_optical(scenes[0].optical_path)  # [3, H, W] uint8 RGB
  sar     = loader.load_sar(scenes[0].sar_path)           # [2, H, W] float32 dB

  # Download (run once)
  python dataset_loader.py --download --dest /data/sen12ms --seasons spring
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SEN12MS constants
# ─────────────────────────────────────────────────────────────────────────────

# ROI numeric ID per season (used in directory and file names)
_ROI_ID: dict[str, str] = {
    "spring": "1158",
    "summer": "1868",
    "fall":   "1970",
    "winter": "2017",
}

ALL_SEASONS = list(_ROI_ID.keys())

# Sentinel-2 band indices (0-based) used for RGB preview and NDVI
_S2_RED   = 3    # B4  665 nm
_S2_GREEN = 2    # B3  560 nm
_S2_BLUE  = 1    # B2  490 nm
_S2_NIR   = 7    # B8  842 nm
_S2_SWIR1 = 10   # B11 1610 nm

# Reflectance scale factor: SEN12MS stores uint16 as reflectance × 10 000
_S2_SCALE = 10_000.0

# Download base URL (TU Munich mediatum data server)
_DOWNLOAD_BASE = "https://dataserv.ub.tum.de/s/m1554803/download?files={filename}"

# Filename of the tar archive per season
_ARCHIVE_NAME: dict[str, str] = {
    "spring": "ROIs1158_spring.tar.gz",
    "summer": "ROIs1868_summer.tar.gz",
    "fall":   "ROIs1970_fall.tar.gz",
    "winter": "ROIs2017_winter.tar.gz",
}

# Simplified 7-class LCLU label set (NDVI/NDWI/NDBI thresholds — see estimate_lclu)
LCLU_CLASSES = (
    "Water Bodies",
    "Forests",
    "Savannas",
    "Croplands",
    "Grasslands",
    "Urban and Built-up Lands",
    "Barren",
)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SceneRecord:
    """One co-registered (optical, SAR) scene pair with geographic metadata."""

    scene_id:     str       # e.g. "ROIs1158_spring_p0001"
    optical_path: Path      # Sentinel-2 GeoTIFF (13 bands)
    sar_path:     Path      # Sentinel-1 GeoTIFF (2 bands: VV, VH)
    season:       str       # spring | summer | fall | winter
    roi_id:       str       # numeric ROI string, e.g. "1158"
    patch_id:     str       # zero-padded patch number, e.g. "0001"
    lat:          float     # WGS84 latitude of scene centre
    lon:          float     # WGS84 longitude of scene centre
    lclu_class:   str       # estimated land cover class


# ─────────────────────────────────────────────────────────────────────────────
# LCLU classification from Sentinel-2 spectral indices
# ─────────────────────────────────────────────────────────────────────────────

def estimate_lclu(s2_bands: np.ndarray) -> str:
    """
    Estimate land cover class from Sentinel-2 bands using spectral indices.

    Falls back to MODIS-label GeoTIFF if available (checked by caller).
    Otherwise applies rule-based classification.

    Thresholds are tuned against the SEN12MS label distribution.

    Args:
        s2_bands: [13, H, W] float64 array (raw uint16 / 10000 already applied).

    Returns:
        String class label from LCLU_CLASSES.
    """
    bands = s2_bands.astype(np.float64)

    nir   = bands[_S2_NIR]
    red   = bands[_S2_RED]
    green = bands[_S2_GREEN]
    swir1 = bands[_S2_SWIR1]

    eps = 1e-6

    # NDVI: vegetation greenness
    ndvi = (nir - red)   / (nir + red   + eps)
    # NDWI: open-water detector (Gao 1996)
    ndwi = (green - nir) / (green + nir + eps)
    # NDBI: built-up / impervious surface index
    ndbi = (swir1 - nir) / (swir1 + nir + eps)

    m_ndvi = float(np.nanmean(ndvi))
    m_ndwi = float(np.nanmean(ndwi))
    m_ndbi = float(np.nanmean(ndbi))

    # Decision tree (IGBP-like, 7 classes)
    if m_ndwi > 0.18:
        return "Water Bodies"
    if m_ndvi > 0.55:
        return "Forests"
    if m_ndvi > 0.35:
        return "Savannas"
    if m_ndvi > 0.20:
        return "Croplands"
    if m_ndbi > 0.10:
        return "Urban and Built-up Lands"
    if m_ndvi > 0.06:
        return "Grasslands"
    return "Barren"


def read_lclu_from_file(lc_path: Path) -> str:
    """
    Read MODIS LCLU class from the companion _lc.tif file when available.

    SEN12MS ships an optional s2_{season}_lc/ sub-directory with per-patch
    MODIS IGBP class rasters (1 band, uint8, values 1–17 or 255).

    Args:
        lc_path: Path to the *_lc.tif file.

    Returns:
        IGBP class name string.
    """
    _IGBP = {
        1: "Evergreen Needleleaf Forests",
        2: "Evergreen Broadleaf Forests",
        3: "Deciduous Needleleaf Forests",
        4: "Deciduous Broadleaf Forests",
        5: "Mixed Forests",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Croplands",
        13: "Urban and Built-up Lands",
        14: "Cropland/Natural Vegetation Mosaics",
        15: "Permanent Snow and Ice",
        16: "Barren",
        17: "Water Bodies",
    }
    try:
        import rasterio
        with rasterio.open(lc_path) as src:
            data = src.read(1).flatten()
        # Majority vote over valid pixels
        valid = data[(data > 0) & (data < 255)]
        if valid.size == 0:
            return "No Data"
        majority = int(np.bincount(valid).argmax())
        return _IGBP.get(majority, f"Class_{majority}")
    except Exception as exc:
        logger.debug("Cannot read LC file %s: %s", lc_path, exc)
        return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Geographic coordinate extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_center_latlon(tif_path: Path) -> tuple[float, float]:
    """
    Compute the WGS84 (lat, lon) of the scene centre from a GeoTIFF.

    Uses rasterio + pyproj to handle any CRS the data might be in.

    Args:
        tif_path: Path to any GeoTIFF in the scene.

    Returns:
        (latitude, longitude) as float, or (0.0, 0.0) on failure.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import transform as rio_transform

        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            cx = (bounds.left + bounds.right)  / 2.0
            cy = (bounds.bottom + bounds.top)  / 2.0
            src_crs = src.crs

        wgs84 = CRS.from_epsg(4326)
        (lon,), (lat,) = rio_transform(src_crs, wgs84, [cx], [cy])
        return round(float(lat), 6), round(float(lon), 6)

    except Exception as exc:
        logger.warning("Cannot extract coordinates from %s: %s", tif_path.name, exc)
        return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

class SEN12MSLoader:
    """
    Discovers and loads co-registered Sentinel-1/2 scene pairs from SEN12MS.

    Parameters
    ----------
    root_dir:   Root directory that contains the ROIs{id}_{season}/ folders.
    seasons:    Subset of ["spring", "summer", "fall", "winter"] to load.
    max_scenes: Hard cap on total number of scenes (for quick tests).
    """

    def __init__(
        self,
        root_dir: str | Path,
        seasons: list[str] | None = None,
        max_scenes: int | None = None,
    ) -> None:
        self.root = Path(root_dir)
        self.seasons = seasons or ALL_SEASONS
        self.max_scenes = max_scenes
        self._scenes: list[SceneRecord] = []

        unknown = set(self.seasons) - set(ALL_SEASONS)
        if unknown:
            raise ValueError(f"Unknown seasons: {unknown}. Choose from {ALL_SEASONS}.")
        if not self.root.exists():
            raise FileNotFoundError(f"SEN12MS root not found: {self.root}")

    # ── Discovery ──────────────────────────────────────────────────────────

    def discover(self, verbose: bool = True) -> list[SceneRecord]:
        """
        Walk root_dir and build a list of all valid (optical, SAR) pairs.

        Only patches that have BOTH an s1 and s2 file are included.
        Results are sorted by scene_id for reproducibility.

        Returns:
            List of SceneRecord (length ≤ max_scenes if set).
        """
        if self._scenes:
            return self._scenes

        records: list[SceneRecord] = []

        for season in self.seasons:
            roi_id = _ROI_ID[season]
            roi_dir = self.root / f"ROIs{roi_id}_{season}"

            if not roi_dir.exists():
                logger.warning("ROI directory not found: %s", roi_dir)
                continue

            s1_dir = roi_dir / f"s1_{season}"
            s2_dir = roi_dir / f"s2_{season}"
            lc_dir = roi_dir / f"s2_{season}_lc"   # optional

            if not s1_dir.exists() or not s2_dir.exists():
                logger.warning("Missing s1 or s2 sub-directory in %s", roi_dir)
                continue

            # Build patch_id → s1 path mapping
            s1_files = self._index_patch_files(s1_dir, f"ROIs{roi_id}_{season}_s1_p")

            paired = 0
            missing = 0

            for patch_id, s1_path in sorted(s1_files.items()):
                s2_path = s2_dir / f"ROIs{roi_id}_{season}_s2_p{patch_id}.tif"
                if not s2_path.exists():
                    missing += 1
                    continue

                # Coordinates from optical file (has better CRS metadata)
                lat, lon = extract_center_latlon(s2_path)

                # LCLU class
                lc_path = lc_dir / f"ROIs{roi_id}_{season}_s2_p{patch_id}_lc.tif"
                if lc_path.exists():
                    lclu = read_lclu_from_file(lc_path)
                else:
                    lclu = self._estimate_lclu_from_file(s2_path)

                scene_id = f"ROIs{roi_id}_{season}_p{patch_id}"

                records.append(SceneRecord(
                    scene_id=scene_id,
                    optical_path=s2_path,
                    sar_path=s1_path,
                    season=season,
                    roi_id=roi_id,
                    patch_id=patch_id,
                    lat=lat,
                    lon=lon,
                    lclu_class=lclu,
                ))
                paired += 1

                if self.max_scenes and len(records) >= self.max_scenes:
                    break

            if verbose:
                logger.info(
                    "Season %-6s  |  paired: %5d  |  SAR-only: %3d",
                    season, paired, missing,
                )

            if self.max_scenes and len(records) >= self.max_scenes:
                break

        self._scenes = records
        if verbose:
            logger.info("Discovery complete — %d paired scenes", len(self._scenes))
        return self._scenes

    # ── Loading ────────────────────────────────────────────────────────────

    def load_optical_rgb(self, path: Path) -> np.ndarray:
        """
        Load Sentinel-2 and return an RGB composite as uint8.

        Extracts bands B4/B3/B2 (Red/Green/Blue), clips to the 2nd–98th
        percentile per band, and scales to [0, 255].

        Args:
            path: Path to the 13-band Sentinel-2 GeoTIFF.

        Returns:
            np.ndarray [3, H, W] uint8 (R, G, B).
        """
        bands = self._read_tif(path)             # [13, H, W] uint16
        rgb = bands[[_S2_RED, _S2_GREEN, _S2_BLUE]].astype(np.float32)

        out = np.empty_like(rgb)
        for i in range(3):
            p2, p98 = np.percentile(rgb[i], (2, 98))
            out[i] = np.clip((rgb[i] - p2) / (p98 - p2 + 1e-8) * 255.0, 0, 255)

        return out.astype(np.uint8)

    def load_optical_all_bands(self, path: Path) -> np.ndarray:
        """
        Load all 13 Sentinel-2 bands as float32 in reflectance units [0, 1].

        Args:
            path: Path to the 13-band Sentinel-2 GeoTIFF.

        Returns:
            np.ndarray [13, H, W] float32, values in [0, ~1].
        """
        return self._read_tif(path).astype(np.float32) / _S2_SCALE

    def load_sar(self, path: Path) -> np.ndarray:
        """
        Load Sentinel-1 SAR bands (VV, VH) as float32 in dB scale.

        SAR values in SEN12MS are stored in linear scale (power ratio).
        This converts: dB = 10 × log₁₀(value) and clips to [-30, 0] dB.

        Args:
            path: Path to the 2-band Sentinel-1 GeoTIFF.

        Returns:
            np.ndarray [2, H, W] float32 (VV, VH) clipped to [-30, 0] dB.
        """
        linear = self._read_tif(path).astype(np.float32)   # [2, H, W]
        db = 10.0 * np.log10(np.clip(linear, 1e-10, None))
        return np.clip(db, -30.0, 0.0)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _read_tif(path: Path) -> np.ndarray:
        """Read all bands from a GeoTIFF → [bands, H, W] ndarray."""
        try:
            import rasterio
            with rasterio.open(path) as src:
                return src.read()                           # [bands, H, W]
        except Exception as exc:
            raise IOError(f"Cannot read GeoTIFF {path}: {exc}") from exc

    @staticmethod
    def _index_patch_files(directory: Path, prefix: str) -> dict[str, Path]:
        """
        Build a {patch_id_str → Path} mapping for all .tif files in a directory.

        Filename pattern: {prefix}NNNN.tif where NNNN is the zero-padded patch ID.
        """
        pattern = re.compile(re.escape(prefix) + r"(\d{4})\.tif$", re.IGNORECASE)
        index: dict[str, Path] = {}
        for p in sorted(directory.glob("*.tif")):
            m = pattern.search(p.name)
            if m:
                index[m.group(1)] = p
        return index

    def _estimate_lclu_from_file(self, s2_path: Path) -> str:
        """Load optical bands and classify LCLU from spectral indices."""
        try:
            bands = self._read_tif(s2_path).astype(np.float64) / _S2_SCALE
            return estimate_lclu(bands)
        except Exception as exc:
            logger.debug("LCLU estimation failed for %s: %s", s2_path.name, exc)
            return "Unknown"

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def scenes(self) -> list[SceneRecord]:
        """Return discovered scenes (call discover() first)."""
        return self._scenes

    def __len__(self) -> int:
        return len(self._scenes)

    def __iter__(self) -> Iterator[SceneRecord]:
        return iter(self._scenes)


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

class SEN12MSDownloader:
    """
    Downloads and extracts SEN12MS seasonal archives.

    Download size per season (approximate):
      spring : ~60 GB
      summer : ~60 GB
      fall   : ~60 GB
      winter : ~60 GB
      total  : ~240 GB (all seasons)

    For the hackathon MVP, downloading only spring (~60 GB) is sufficient
    to demonstrate cross-modal retrieval on ~45 000 paired scenes.
    """

    def __init__(self, dest_dir: str | Path) -> None:
        self.dest = Path(dest_dir)
        self.dest.mkdir(parents=True, exist_ok=True)

    def download(self, seasons: list[str] | None = None) -> None:
        """
        Download and extract the requested seasonal archives.

        Args:
            seasons: Subset of ALL_SEASONS. Defaults to ["spring"].
        """
        seasons = seasons or ["spring"]
        for season in seasons:
            archive_name = _ARCHIVE_NAME[season]
            archive_path = self.dest / archive_name
            roi_dir = self.dest / f"ROIs{_ROI_ID[season]}_{season}"

            if roi_dir.exists():
                logger.info("Season '%s' already extracted at %s — skipping.", season, roi_dir)
                continue

            if not archive_path.exists():
                url = _DOWNLOAD_BASE.format(filename=archive_name)
                self._download_file(url, archive_path)
            else:
                logger.info("Archive already downloaded: %s", archive_path)

            logger.info("Extracting %s …", archive_path)
            self._extract(archive_path, self.dest)
            logger.info("Extracted to %s", roi_dir)

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        """Download a single file with progress reporting."""
        logger.info("Downloading %s → %s", url, dest)

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded / total_size * 100)
                gb = downloaded / 1e9
                print(f"\r  {pct:5.1f}%  ({gb:.2f} GB)", end="", flush=True)

        try:
            urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
            print()  # newline after progress
        except Exception as exc:
            raise RuntimeError(
                f"Download failed: {exc}\n"
                f"URL: {url}\n"
                f"Manual download: https://mediatum.ub.tum.de/1554803"
            ) from exc

    @staticmethod
    def _extract(archive: Path, dest: Path) -> None:
        """Extract a .tar.gz archive into dest/."""
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=str(dest))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (download mode)
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SEN12MS Dataset Loader — discover or download SEN12MS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # Download spring season into /data/sen12ms
  python dataset_loader.py --download --dest /data/sen12ms --seasons spring

  # Validate discovery of existing download
  python dataset_loader.py --root /data/sen12ms --seasons spring summer
""",
    )
    p.add_argument("--root",    type=str, help="SEN12MS root directory (for --discover)")
    p.add_argument("--dest",    type=str, help="Download destination directory")
    p.add_argument("--seasons", nargs="+", default=["spring"], choices=ALL_SEASONS)
    p.add_argument("--download",  action="store_true", help="Download and extract archives")
    p.add_argument("--max-scenes", type=int, default=None, help="Cap total scenes discovered")
    p.add_argument("--verbose",   action="store_true")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    args = _build_parser().parse_args()

    if args.download:
        if not args.dest:
            sys.exit("--dest required with --download")
        dl = SEN12MSDownloader(args.dest)
        dl.download(args.seasons)
        root = args.dest
    elif args.root:
        root = args.root
    else:
        sys.exit("Provide --root (existing data) or --download --dest (new download).")

    loader = SEN12MSLoader(root, seasons=args.seasons, max_scenes=args.max_scenes)
    scenes = loader.discover(verbose=True)

    print(f"\nDiscovered {len(scenes)} paired scenes.")
    if scenes:
        s = scenes[0]
        print(f"First scene  :  {s.scene_id}")
        print(f"  Optical    :  {s.optical_path}")
        print(f"  SAR        :  {s.sar_path}")
        print(f"  Coords     :  lat={s.lat}, lon={s.lon}")
        print(f"  LCLU       :  {s.lclu_class}")
        print(f"  Season     :  {s.season}")


if __name__ == "__main__":
    main()
