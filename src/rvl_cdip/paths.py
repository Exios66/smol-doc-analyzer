"""Filesystem layout for RVL-CDIP downloads and the derived SQLite DB.

All Hub downloads and the SQL database live under the project ``.venv`` tree
so the ~38 GB image archive never lands in ``data/`` or the user home cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import REPO_ROOT

# Public Hub dataset (label files + archive live under data/ on this repo).
HF_DATASET_ID = "aharley/rvl_cdip"
HF_DATA_REPO_FALLBACK = "rvl_cdip"  # legacy resolve URLs used by the dataset script

LABEL_FILES = {
    "train": "data/train.txt",
    "test": "data/test.txt",
    "validation": "data/val.txt",
}

IMAGE_ARCHIVE_NAME = "rvl-cdip.tar.gz"
IMAGE_ARCHIVE_REMOTE = f"data/{IMAGE_ARCHIVE_NAME}"

# ~38.76 GB compressed archive (from dataset_infos.json). Used for preflight checks.
IMAGE_ARCHIVE_BYTES_ESTIMATE = 38_762_320_458
# Minimum free space recommended before starting the image download (archive + headroom).
IMAGE_DOWNLOAD_MIN_FREE_BYTES = 45 * 1024**3

LABEL_NAMES: tuple[str, ...] = (
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
)


def venv_root() -> Path:
    """Project virtualenv root (``.venv``). Created on demand."""
    root = REPO_ROOT / ".venv"
    root.mkdir(parents=True, exist_ok=True)
    return root


def rvl_root() -> Path:
    """Root for all RVL-CDIP artifacts: ``.venv/rvl_cdip/``.

    Always derived from the project ``.venv`` (never home-cache or ``data/``).
    ``RVL_CDIP_ROOT`` / Config may point the SQLite file elsewhere under
    ``.venv`` via :func:`default_db_path`, but source downloads stay here.
    """
    path = venv_root() / "rvl_cdip"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_home() -> Path:
    """Hugging Face cache home forced under ``.venv`` (never ``~/.cache``)."""
    path = rvl_root() / "hf_home"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_hub_cache() -> Path:
    path = hf_home() / "hub"
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_dir() -> Path:
    """Downloaded label files / archive live here (source for SQL)."""
    path = rvl_root() / "source"
    path.mkdir(parents=True, exist_ok=True)
    return path


def labels_dir() -> Path:
    path = source_dir() / "labels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_path() -> Path:
    return source_dir() / IMAGE_ARCHIVE_NAME


def images_dir() -> Path:
    """Extracted TIFF images (optional; only after explicit archive extract)."""
    path = source_dir() / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_db_path() -> Path:
    """SQLite database path under ``.venv/rvl_cdip/``."""
    try:
        from src.utils.config import Config

        path = Path(Config.load().rvl_cdip_db_path)
    except Exception:
        path = rvl_root() / "rvl_cdip.db"
    path = assert_path_under_venv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def apply_hf_cache_env() -> dict[str, str]:
    """Force Hub SDKs to read/write only under ``.venv/rvl_cdip/hf_home``.

    Returns the env keys that were set (for logging / tests).
    """
    home = str(hf_home())
    hub = str(hf_hub_cache())
    # HF_HOME is the primary knob; also pin the hub cache and datasets cache
    # so neither huggingface_hub nor datasets can escape to ~/.cache.
    os.environ["HF_HOME"] = home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub
    os.environ["HF_HUB_CACHE"] = hub
    os.environ["HF_DATASETS_CACHE"] = str(hf_home() / "datasets")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return {
        "HF_HOME": home,
        "HUGGINGFACE_HUB_CACHE": hub,
        "HF_HUB_CACHE": hub,
        "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
    }


def assert_path_under_venv(path: Path) -> Path:
    """Raise if ``path`` is not inside the project ``.venv`` tree."""
    resolved = path.resolve()
    venv = venv_root().resolve()
    try:
        resolved.relative_to(venv)
    except ValueError as exc:
        raise RuntimeError(
            f"RVL-CDIP path must stay under {venv}; refused: {resolved}"
        ) from exc
    return resolved
