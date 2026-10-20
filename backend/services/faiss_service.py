"""
GeoSense-X — FAISS Index Service
==================================
Manages three IVFFlat FAISS indexes and executes top-K similarity search.

Index layout
────────────
  optical.index  →  N optical embeddings       (FAISS ID i  →  scenes[i], optical)
  sar.index      →  N SAR embeddings           (FAISS ID i  →  scenes[i], sar)
  shared.index   →  2N embeddings (all mods)   (FAISS ID i  →  decode_shared_id(i))
                    IDs  0 … N-1  : optical
                    IDs  N … 2N-1 : SAR

id_map.json format
──────────────────
  {
    "scenes": ["scene_001", "scene_002", …],   ← length N, same order as *.npy files
    "n": N
  }

Building indexes (run once, outside of FastAPI)
──────────────────────────────────────────────
  from services.faiss_service import build_indexes
  build_indexes(
      embeddings_dir=Path("embeddings"),   # contains optical.npy, sar.npy, metadata.json
      index_dir=Path("indexes"),
  )

Requirements
────────────
  embeddings/optical.npy   → [N, D]  float32, L2-normalised
  embeddings/sar.npy        → [N, D]  float32, L2-normalised
  embeddings/metadata.json  → list of N dicts with at minimum {"scene_id": "..."}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single FAISS search hit decoded to business fields."""
    scene_id: str
    modality: str
    similarity: float
    metadata: dict = field(default_factory=dict)


class FAISSService:
    """
    Loads FAISS indexes at startup and provides a unified search() interface.

    Thread-safe for reads: FAISS IndexFlatIP is read-only after load().
    Stored once in app.state.faiss and reused across all requests.
    """

    def __init__(self, index_dir: Path, metadata_path: Path) -> None:
        self.index_dir = Path(index_dir)
        self.metadata_path = Path(metadata_path)
        self._indexes: dict[str, faiss.Index] = {}
        self._scenes: list[str] = []       # scene_id ordered list (length N)
        self._n: int = 0                   # N = scenes per modality
        self._metadata: dict[str, dict] = {}  # scene_id → full metadata dict

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load all indexes and metadata from disk. Called once at startup."""
        self._load_metadata()
        for name in ("optical", "sar", "shared"):
            path = self.index_dir / f"{name}.index"
            if path.exists():
                self._indexes[name] = faiss.read_index(str(path))
                logger.info(
                    "Loaded index '%s' — %d vectors (d=%d)",
                    name,
                    self._indexes[name].ntotal,
                    self._indexes[name].d,
                )
            else:
                logger.warning("Index not found (will be skipped): %s", path)

        if not self._indexes:
            logger.warning(
                "No FAISS indexes found in '%s'. "
                "Run build_indexes() after generating embeddings.",
                self.index_dir,
            )

    def _load_metadata(self) -> None:
        """Load scene list and id_map from disk."""
        id_map_path = self.index_dir / "id_map.json"
        if id_map_path.exists():
            with open(id_map_path) as f:
                data = json.load(f)
            self._scenes = data.get("scenes", [])
            self._n = data.get("n", len(self._scenes))
        else:
            logger.warning("id_map.json not found in %s", self.index_dir)

        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                records = json.load(f)
            self._metadata = {r["scene_id"]: r for r in records if "scene_id" in r}
            logger.info("Loaded metadata for %d scenes", len(self._metadata))
        else:
            logger.warning("metadata.json not found at %s", self.metadata_path)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        embedding: np.ndarray,
        source_modality: str,
        target_modality: str,
        k: int = 10,
    ) -> list[SearchResult]:
        """
        Find the top-K most similar scenes.

        Routing:
          source == target  →  per-modality index (optical.index or sar.index)
          target == "all"   →  shared.index, no modality filter
          source != target  →  shared.index, filter to target modality

        Args:
            embedding:        [D] float32, L2-normalised.
            source_modality:  "optical" | "sar"
            target_modality:  "optical" | "sar" | "all"
            k:                Number of results to return.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        if not self._indexes:
            logger.error("No indexes loaded — cannot search")
            return []

        query = embedding.reshape(1, -1).astype(np.float32)

        # Same-modal retrieval
        if source_modality == target_modality and source_modality in self._indexes:
            return self._search_per_modality(query, source_modality, k)

        # Cross-modal or all-modal (need shared index)
        if "shared" in self._indexes:
            filter_mod = None if target_modality == "all" else target_modality
            return self._search_shared(query, filter_mod, k)

        # Fallback: if no shared index, use whatever per-modality index exists
        for name in self._indexes:
            if name != "shared":
                logger.warning(
                    "Shared index unavailable. Falling back to '%s' index.", name
                )
                return self._search_per_modality(query, name, k)

        return []

    def _search_per_modality(
        self, query: np.ndarray, modality: str, k: int
    ) -> list[SearchResult]:
        """Search a single per-modality index."""
        index = self._indexes[modality]
        n_results = min(k, index.ntotal)
        if n_results == 0:
            return []

        D, I = index.search(query, n_results)
        results: list[SearchResult] = []
        for sim, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            scene_id = self._scenes[idx] if idx < len(self._scenes) else f"scene_{idx}"
            results.append(SearchResult(
                scene_id=scene_id,
                modality=modality,
                similarity=float(np.clip(sim, 0.0, 1.0)),
                metadata=self._metadata.get(scene_id, {}),
            ))
        return results

    def _search_shared(
        self, query: np.ndarray, target_modality: str | None, k: int
    ) -> list[SearchResult]:
        """
        Search the shared index (all modalities) and optionally filter to target.

        Over-fetches by 5× when filtering to ensure we get k results after filter.
        """
        index = self._indexes["shared"]
        fetch_k = min(k * 5 if target_modality else k, index.ntotal)
        if fetch_k == 0:
            return []

        D, I = index.search(query, fetch_k)
        results: list[SearchResult] = []

        for sim, faiss_id in zip(D[0], I[0]):
            if faiss_id < 0:
                continue

            scene_id, modality = self._decode_shared_id(int(faiss_id))

            if target_modality and modality != target_modality:
                continue

            results.append(SearchResult(
                scene_id=scene_id,
                modality=modality,
                similarity=float(np.clip(sim, 0.0, 1.0)),
                metadata=self._metadata.get(scene_id, {}),
            ))

            if len(results) >= k:
                break

        return results

    def _decode_shared_id(self, faiss_id: int) -> tuple[str, str]:
        """
        Decode a shared-index FAISS integer ID to (scene_id, modality).

        IDs 0 … N-1  → optical
        IDs N … 2N-1 → SAR
        """
        n = self._n or (len(self._scenes) if self._scenes else 1)
        if faiss_id < n:
            scene_id = self._scenes[faiss_id] if faiss_id < len(self._scenes) else f"scene_{faiss_id}"
            return scene_id, "optical"
        else:
            idx = faiss_id - n
            scene_id = self._scenes[idx] if idx < len(self._scenes) else f"scene_{idx}"
            return scene_id, "sar"

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def ready(self) -> bool:
        """True if at least one index is loaded."""
        return bool(self._indexes)

    def status(self) -> dict:
        """Return index sizes for the /health endpoint."""
        return {
            name: {"ntotal": idx.ntotal, "d": idx.d}
            for name, idx in self._indexes.items()
        }


# ────────────────────────────────────────────────────────────────────────────
# Index builder — run once from the command line after computing embeddings
# ────────────────────────────────────────────────────────────────────────────

def build_indexes(embeddings_dir: Path, index_dir: Path) -> None:
    """
    Build FAISS IVFFlat indexes from pre-computed embedding numpy arrays.

    Reads
    ─────
      {embeddings_dir}/optical.npy    [N, D] float32  (L2-normalised)
      {embeddings_dir}/sar.npy        [N, D] float32  (L2-normalised)
      {embeddings_dir}/metadata.json  list of N dicts  {"scene_id": …, …}

    Writes
    ──────
      {index_dir}/optical.index
      {index_dir}/sar.index
      {index_dir}/shared.index
      {index_dir}/id_map.json

    Usage (from the project root)
    ─────────────────────────────
      python - <<'EOF'
      from pathlib import Path
      from services.faiss_service import build_indexes
      build_indexes(Path("embeddings"), Path("indexes"))
      EOF
    """
    embeddings_dir = Path(embeddings_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings ──────────────────────────────────────────────────────
    opt_path = embeddings_dir / "optical.npy"
    sar_path = embeddings_dir / "sar.npy"
    meta_path = embeddings_dir / "metadata.json"

    if not opt_path.exists():
        raise FileNotFoundError(f"optical.npy not found at {opt_path}")
    if not sar_path.exists():
        raise FileNotFoundError(f"sar.npy not found at {sar_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")

    logger.info("Loading embeddings …")
    opt_embs = np.load(str(opt_path)).astype(np.float32)   # [N, D]
    sar_embs = np.load(str(sar_path)).astype(np.float32)   # [N, D]

    with open(meta_path) as f:
        metadata = json.load(f)

    n, d = opt_embs.shape
    assert sar_embs.shape == (n, d), (
        f"optical.npy has {n} rows but sar.npy has {sar_embs.shape[0]} rows. "
        f"Both must have the same number of scenes in the same order."
    )
    assert len(metadata) == n, (
        f"metadata.json has {len(metadata)} entries but embeddings have {n} rows."
    )

    logger.info("  %d scenes, %d-d embeddings", n, d)

    # L2 normalise (in-place) — ensures dot product == cosine similarity
    faiss.normalize_L2(opt_embs)
    faiss.normalize_L2(sar_embs)

    # Per-modality indexes ─────────────────────────────────────────────────
    for name, embs in (("optical", opt_embs), ("sar", sar_embs)):
        index = faiss.IndexFlatIP(d)        # exact inner-product search
        index.add(embs)
        out = str(index_dir / f"{name}.index")
        faiss.write_index(index, out)
        logger.info("  Saved %s (%d vectors)", out, index.ntotal)

    # Shared index (optical first, then SAR) ───────────────────────────────
    all_embs = np.vstack([opt_embs, sar_embs])   # [2N, D]
    shared = faiss.IndexFlatIP(d)
    shared.add(all_embs)
    out_shared = str(index_dir / "shared.index")
    faiss.write_index(shared, out_shared)
    logger.info("  Saved %s (%d vectors)", out_shared, shared.ntotal)

    # id_map ───────────────────────────────────────────────────────────────
    # Per-modality: FAISS ID i → metadata[i]["scene_id"]
    # Shared:       FAISS ID i < N → optical, ID i >= N → SAR
    scene_ids = [record["scene_id"] for record in metadata]
    id_map = {"scenes": scene_ids, "n": n}

    out_map = index_dir / "id_map.json"
    with open(out_map, "w") as f:
        json.dump(id_map, f)
    logger.info("  Saved %s", out_map)

    logger.info(
        "build_indexes complete — %d scenes, %d-d, 3 indexes written to %s",
        n, d, index_dir,
    )
