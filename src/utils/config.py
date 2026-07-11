"""
Centralized configuration, loaded from environment variables / .env.

Nothing in this file should contain secrets directly -- values are read
from the environment so .env (gitignored) is the only place real keys live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _path(env_var: str, default: str) -> Path:
    raw = os.getenv(env_var, default)
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    generation_model: str
    max_concurrency: int
    max_retries: int

    skeleton_output_dir: Path
    document_output_dir: Path
    memo_output_dir: Path
    provenance_log_path: Path

    taxonomy_path: Path
    claim_schema_path: Path

    @classmethod
    def load(cls) -> "Config":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            generation_model=os.getenv("GENERATION_MODEL", "claude-sonnet-5"),
            max_concurrency=int(os.getenv("GENERATION_MAX_CONCURRENCY", "4")),
            max_retries=int(os.getenv("GENERATION_MAX_RETRIES", "5")),
            skeleton_output_dir=_path("SKELETON_OUTPUT_DIR", "data/synthetic/skeletons"),
            document_output_dir=_path("DOCUMENT_OUTPUT_DIR", "data/synthetic/documents"),
            memo_output_dir=_path("MEMO_OUTPUT_DIR", "data/synthetic/memos"),
            provenance_log_path=_path("PROVENANCE_LOG_PATH", "data/provenance_log.jsonl"),
            taxonomy_path=REPO_ROOT / "taxonomy" / "acord_form_categories.yaml",
            claim_schema_path=REPO_ROOT / "data" / "schemas" / "claim_skeleton.schema.json",
        )

    def ensure_dirs(self) -> None:
        for d in (self.skeleton_output_dir, self.document_output_dir, self.memo_output_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.provenance_log_path.parent.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    cfg = Config.load()
    cfg.ensure_dirs()
    return cfg
