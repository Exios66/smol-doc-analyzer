"""Project configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


@dataclass(frozen=True)
class Config:
    openrouter_api_key: str
    generation_model: str
    max_concurrency: int
    max_retries: int
    openrouter_app_url: str
    openrouter_app_name: str

    skeleton_output_dir: Path
    document_output_dir: Path
    memo_output_dir: Path
    noisy_output_dir: Path
    provenance_log_path: Path

    taxonomy_path: Path
    claim_schema_path: Path
    profiles_dir: Path
    raw_data_dir: Path
    splits_path: Path
    evaluation_reports_dir: Path
    models_dir: Path

    wandb_project: str
    wandb_entity: str
    wandb_mode: str
    wandb_api_key: str

    @classmethod
    def load(cls) -> "Config":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            generation_model=os.getenv("GENERATION_MODEL", "anthropic/claude-sonnet-4.5"),
            max_concurrency=int(os.getenv("GENERATION_MAX_CONCURRENCY", "4")),
            max_retries=int(os.getenv("GENERATION_MAX_RETRIES", "5")),
            openrouter_app_url=os.getenv("OPENROUTER_APP_URL", ""),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", ""),
            skeleton_output_dir=_path("SKELETON_OUTPUT_DIR", "data/synthetic/skeletons"),
            document_output_dir=_path("DOCUMENT_OUTPUT_DIR", "data/synthetic/documents"),
            memo_output_dir=_path("MEMO_OUTPUT_DIR", "data/synthetic/memos"),
            noisy_output_dir=_path("NOISY_OUTPUT_DIR", "data/synthetic/noisy"),
            provenance_log_path=_path("PROVENANCE_LOG_PATH", "data/provenance_log.jsonl"),
            taxonomy_path=REPO_ROOT / "taxonomy" / "acord_form_categories.yaml",
            claim_schema_path=REPO_ROOT / "data" / "schemas" / "claim_skeleton.schema.json",
            profiles_dir=_path("PROFILES_DIR", "data/profiles"),
            raw_data_dir=_path("RAW_DATA_DIR", "data/raw"),
            splits_path=_path("SPLITS_PATH", "data/synthetic/splits.json"),
            evaluation_reports_dir=_path("EVALUATION_REPORTS_DIR", "evaluation/reports"),
            models_dir=_path("MODELS_DIR", "models"),
            wandb_project=os.getenv("WANDB_PROJECT", "smol-doc-analyzer"),
            wandb_entity=os.getenv("WANDB_ENTITY", ""),
            wandb_mode=os.getenv("WANDB_MODE", "online"),
            wandb_api_key=os.getenv("WANDB_API_KEY", ""),
        )
