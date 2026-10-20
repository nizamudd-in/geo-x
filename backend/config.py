"""
GeoSense-X — Application Configuration
All settings read from environment variables or .env file.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    # Available: dinov2_vits14 (384-d), dinov2_vitb14 (768-d), dinov2_vitl14 (1024-d)
    DINOV2_MODEL: str = "dinov2_vitb14"
    EMBEDDING_DIM: int = 768          # must match the model above
    DEVICE: str = "cpu"               # "cpu" | "cuda" | "mps"

    # ── Paths ──────────────────────────────────────────────────────────────
    FAISS_INDEX_DIR: Path = Path("indexes")
    EMBEDDINGS_DIR: Path = Path("embeddings")
    METADATA_PATH: Path = Path("embeddings/metadata.json")
    THUMBNAILS_DIR: Path = Path("static/thumbnails")

    # ── Search ─────────────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 10
    MAX_FILE_SIZE_MB: int = 50

    # ── Server ─────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    @model_validator(mode="after")
    def validate_dim_matches_model(self) -> "Settings":
        """Warn if EMBEDDING_DIM doesn't match the chosen DINOv2 variant."""
        expected = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        variant = self.DINOV2_MODEL
        if variant in expected and expected[variant] != self.EMBEDDING_DIM:
            raise ValueError(
                f"{variant} outputs {expected[variant]}-d embeddings, "
                f"but EMBEDDING_DIM is set to {self.EMBEDDING_DIM}. "
                f"Set EMBEDDING_DIM={expected[variant]} in your .env file."
            )
        return self


settings = Settings()
