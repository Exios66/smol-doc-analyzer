"""Queryable SQLite index for the public RVL-CDIP document dataset.

Downloads are confined to ``.venv/rvl_cdip/`` (HF cache + label/source files +
SQLite DB). The default build uses only the small split label files; the
~38 GB image archive is opt-in.
"""

from src.rvl_cdip.paths import HF_DATASET_ID, LABEL_NAMES, default_db_path, rvl_root
from src.rvl_cdip.schema import SCHEMA_VERSION
from src.rvl_cdip.store import RvlCdipStore

__all__ = [
    "HF_DATASET_ID",
    "LABEL_NAMES",
    "SCHEMA_VERSION",
    "RvlCdipStore",
    "default_db_path",
    "rvl_root",
]
