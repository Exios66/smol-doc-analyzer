"""Project configuration loaded from environment variables and optional `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Values that look set but are only documentation placeholders.
_PLACEHOLDER_SECRETS = {
    "",
    "sk-or-v1-your-key-here",
    "your-key-here",
    "changeme",
    "replace-me",
    "<your-openrouter-api-key>",
    "<your-wandb-api-key>",
    "<your-hf-token>",
}


def _load_dotenv() -> Path | None:
    """Load repo-root `.env` into os.environ without overriding real env vars."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Fallback parser so a missing optional install still works in a pinch.
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
        return env_path

    load_dotenv(env_path, override=False)
    return env_path


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _bool(env_key: str, default: bool = False) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _secret(env_key: str, default: str = "") -> str:
    """Read a secret env var; treat documentation placeholders as unset."""
    raw = os.getenv(env_key, default)
    if raw is None:
        return ""
    value = raw.strip()
    if value.lower() in {p.lower() for p in _PLACEHOLDER_SECRETS}:
        return ""
    return value


def secrets_status() -> dict[str, bool]:
    """Return whether each known secret is present (never returns the values)."""
    _load_dotenv()
    return {
        "OPENROUTER_API_KEY": bool(_secret("OPENROUTER_API_KEY")),
        "WANDB_API_KEY": bool(_secret("WANDB_API_KEY")),
        "HF_TOKEN": bool(_secret("HF_TOKEN") or _secret("HUGGING_FACE_HUB_TOKEN")),
        "DISCORD_TOKEN": bool(_secret("DISCORD_TOKEN")),
        "DISCORD_WEBHOOK_URL": bool(_secret("DISCORD_WEBHOOK_URL")),
        "DISCORD_AI_API_KEY": bool(
            _secret("DISCORD_AI_API_KEY")
            or _secret("AI_API_KEY")
            or _secret("OPENROUTER_API_KEY")
        ),
        "VISION_LLM_MODEL_PATH": bool(os.getenv("VISION_LLM_MODEL_PATH", "").strip()),
        "SUMMARIZER_MODEL_PATH": bool(os.getenv("SUMMARIZER_MODEL_PATH", "").strip()),
        "dotenv_file": (REPO_ROOT / ".env").exists(),
    }


@dataclass(frozen=True)
class Config:
    openrouter_api_key: str
    generation_model: str
    max_concurrency: int
    max_retries: int
    openrouter_app_url: str
    openrouter_app_name: str
    # When paid OpenRouter credits are exhausted (HTTP 402), route to these free models.
    openrouter_free_fallback_models: tuple[str, ...]
    # Start on free models immediately (skip paid GENERATION_MODEL).
    openrouter_prefer_free: bool

    skeleton_output_dir: Path
    document_output_dir: Path
    memo_output_dir: Path
    noisy_output_dir: Path
    provenance_log_path: Path
    sample_corpus_dir: Path
    sample_corpus_db_path: Path

    taxonomy_path: Path
    claim_schema_path: Path
    profiles_dir: Path
    raw_data_dir: Path
    splits_path: Path
    evaluation_reports_dir: Path
    models_dir: Path

    # Chained inference pipeline (Phase 5)
    pipeline_output_dir: Path
    pipeline_cache_dir: Path
    vision_llm_enabled: bool
    vision_llm_model: str
    vision_llm_model_path: Path | None
    summarizer_model: str
    summarizer_model_path: Path | None

    # Weights & Biases experiment tracking
    wandb_project: str
    wandb_entity: str
    wandb_mode: str
    wandb_api_key: str

    # Optional Hugging Face Hub token (model/dataset downloads)
    hf_token: str

    @classmethod
    def load(cls) -> "Config":
        _load_dotenv()

        vision_path = os.getenv("VISION_LLM_MODEL_PATH", "").strip()
        summarizer_path = os.getenv("SUMMARIZER_MODEL_PATH", "").strip()
        openrouter_key = _secret("OPENROUTER_API_KEY")
        wandb_key = _secret("WANDB_API_KEY")
        hf_token = _secret("HF_TOKEN") or _secret("HUGGING_FACE_HUB_TOKEN")

        # Propagate secrets so third-party SDKs (wandb, huggingface_hub) see them.
        # Also clear placeholder values left in os.environ by .env templates.
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
        else:
            os.environ.pop("WANDB_API_KEY", None)
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

        free_raw = os.getenv("OPENROUTER_FREE_FALLBACK_MODELS", "").strip()
        if free_raw:
            free_fallbacks = tuple(m.strip() for m in free_raw.split(",") if m.strip())
        else:
            # Keep in sync with src.utils.llm_client.DEFAULT_FREE_FALLBACK_MODELS
            free_fallbacks = (
                "openrouter/free",
                "meta-llama/llama-3.2-3b-instruct:free",
                "openai/gpt-oss-20b:free",
            )
        prefer_free = any(
            os.getenv(k, "").strip().lower() in {"1", "true", "yes", "on"}
            for k in ("OPENROUTER_PREFER_FREE", "GENERATION_PREFER_FREE")
        )

        return cls(
            openrouter_api_key=openrouter_key,
            generation_model=os.getenv("GENERATION_MODEL", "anthropic/claude-sonnet-4.5"),
            max_concurrency=int(os.getenv("GENERATION_MAX_CONCURRENCY", "4")),
            max_retries=int(os.getenv("GENERATION_MAX_RETRIES", "5")),
            openrouter_app_url=os.getenv("OPENROUTER_APP_URL", ""),
            openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", ""),
            openrouter_free_fallback_models=free_fallbacks,
            openrouter_prefer_free=prefer_free,
            skeleton_output_dir=_path("SKELETON_OUTPUT_DIR", "data/synthetic/skeletons"),
            document_output_dir=_path("DOCUMENT_OUTPUT_DIR", "data/synthetic/documents"),
            memo_output_dir=_path("MEMO_OUTPUT_DIR", "data/synthetic/memos"),
            noisy_output_dir=_path("NOISY_OUTPUT_DIR", "data/synthetic/noisy"),
            provenance_log_path=_path("PROVENANCE_LOG_PATH", "data/provenance_log.jsonl"),
            sample_corpus_dir=_path("SAMPLE_CORPUS_DIR", "data/sample_corpus"),
            sample_corpus_db_path=_path(
                "SAMPLE_CORPUS_DB_PATH", "data/sample_corpus/documents.db"
            ),
            taxonomy_path=REPO_ROOT / "taxonomy" / "acord_form_categories.yaml",
            claim_schema_path=REPO_ROOT / "data" / "schemas" / "claim_skeleton.schema.json",
            profiles_dir=_path("PROFILES_DIR", "data/profiles"),
            raw_data_dir=_path("RAW_DATA_DIR", "data/raw"),
            splits_path=_path("SPLITS_PATH", "data/synthetic/splits.json"),
            evaluation_reports_dir=_path("EVALUATION_REPORTS_DIR", "evaluation/reports"),
            models_dir=_path("MODELS_DIR", "models"),
            pipeline_output_dir=_path("PIPELINE_OUTPUT_DIR", "data/pipeline"),
            pipeline_cache_dir=_path("PIPELINE_CACHE_DIR", "data/pipeline/cache"),
            vision_llm_enabled=_bool("VISION_LLM_ENABLED", default=True),
            vision_llm_model=os.getenv(
                "VISION_LLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct"
            ),
            vision_llm_model_path=_path("VISION_LLM_MODEL_PATH", vision_path)
            if vision_path
            else None,
            summarizer_model=os.getenv("SUMMARIZER_MODEL", ""),
            summarizer_model_path=_path("SUMMARIZER_MODEL_PATH", summarizer_path)
            if summarizer_path
            else None,
            wandb_project=os.getenv("WANDB_PROJECT", "smol-doc-analyzer"),
            wandb_entity=os.getenv("WANDB_ENTITY", ""),
            wandb_mode=os.getenv("WANDB_MODE", "online"),
            wandb_api_key=wandb_key,
            hf_token=hf_token,
        )
