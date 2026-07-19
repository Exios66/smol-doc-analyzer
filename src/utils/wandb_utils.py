"""Weights & Biases helpers for training, evaluation, and pipeline runs.

Logging is opt-out: enabled when ``wandb`` is installed and not disabled via
``WANDB_MODE=disabled``, ``WANDB_DISABLED=true``, or an explicit ``enabled=False``.
Without an API key, runs fall back to offline mode so local smoke tests still work.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool
    project: str
    entity: str | None
    mode: str  # online | offline | disabled
    tags: tuple[str, ...] = ()


def _wandb_netrc_available() -> bool:
    """True when ``wandb login`` credentials exist for api.wandb.ai."""
    try:
        import netrc

        auth = netrc.netrc().authenticators("api.wandb.ai")
        return bool(auth and (auth[2] or auth[0]))
    except Exception:  # noqa: BLE001
        return False


def load_wandb_settings(
    *,
    enabled: bool | None = None,
    project: str | None = None,
    entity: str | None = None,
    mode: str | None = None,
    tags: Sequence[str] | None = None,
) -> WandbSettings:
    """Resolve WandB settings from explicit overrides and environment."""
    from src.utils.config import _load_dotenv, _secret

    # CLI may call this before Config.load(); ensure repo .env is visible.
    _load_dotenv()

    env_mode = (mode or os.getenv("WANDB_MODE", "online")).strip().lower()
    disabled_flag = os.getenv("WANDB_DISABLED", "").strip().lower() in {"1", "true", "yes"}
    if env_mode == "disabled" or disabled_flag:
        resolved_enabled = False
        env_mode = "disabled"
    elif enabled is None:
        resolved_enabled = True
    else:
        resolved_enabled = enabled

    # Treat documentation placeholders as unset (Config._secret already does this
    # for the Config object, but wandb reads os.environ directly).
    api_key = _secret("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    elif os.getenv("WANDB_API_KEY") is not None and not os.getenv("WANDB_API_KEY", "").strip():
        # Empty .env entries should not block wandb login / netrc auth.
        os.environ.pop("WANDB_API_KEY", None)
    elif os.getenv("WANDB_API_KEY"):
        # Clear placeholders so the wandb SDK does not attempt an online login.
        os.environ.pop("WANDB_API_KEY", None)

    has_login = _wandb_netrc_available()
    if resolved_enabled and env_mode == "online" and not api_key and not has_login:
        logger.info(
            "WANDB_API_KEY unset and no wandb login found; using offline mode"
        )
        env_mode = "offline"
    elif resolved_enabled and env_mode == "online" and not api_key and has_login:
        logger.info("Using wandb login credentials (no WANDB_API_KEY in .env)")

    return WandbSettings(
        enabled=resolved_enabled and env_mode != "disabled",
        project=(project or os.getenv("WANDB_PROJECT") or "smol-doc-analyzer").strip(),
        entity=(entity if entity is not None else os.getenv("WANDB_ENTITY") or "").strip() or None,
        mode=env_mode if resolved_enabled else "disabled",
        tags=tuple(tags or ()),
    )


class WandbRun:
    """Thin wrapper so callers never need to import wandb directly."""

    def __init__(self, run: Any | None = None, settings: WandbSettings | None = None):
        self._run = run
        self.settings = settings or WandbSettings(
            enabled=False, project="smol-doc-analyzer", entity=None, mode="disabled"
        )

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def report_to(self) -> list[str]:
        """Value for Hugging Face ``TrainingArguments.report_to``."""
        return ["wandb"] if self.active else []

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if not self.active:
            return
        if step is None:
            self._run.log(data)
        else:
            self._run.log(data, step=step)

    def summary(self, data: dict[str, Any]) -> None:
        if not self.active:
            return
        for key, value in data.items():
            self._run.summary[key] = value

    def log_table(
        self,
        key: str,
        columns: Sequence[str],
        data: Sequence[Sequence[Any]],
    ) -> None:
        if not self.active:
            return
        import wandb

        try:
            table = wandb.Table(columns=list(columns), data=[list(row) for row in data])
            self._run.log({key: table})
        except Exception as exc:  # noqa: BLE001 — tables are best-effort
            logger.warning("Could not log WandB table %s: %s", key, exc)

    def log_confusion_matrix(
        self,
        *,
        key: str,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        class_names: Sequence[str],
    ) -> None:
        if not self.active:
            return
        import wandb

        try:
            self._run.log(
                {
                    key: wandb.plot.confusion_matrix(
                        y_true=list(y_true),
                        preds=list(y_pred),
                        class_names=list(class_names),
                    )
                }
            )
        except Exception as exc:  # noqa: BLE001 — plotting is best-effort
            logger.warning("Could not log WandB confusion matrix: %s", exc)
            # Fallback: flat table of true/pred counts
            from collections import Counter

            counts = Counter(zip(y_true, y_pred))
            rows = [
                [class_names[i], class_names[j], c]
                for (i, j), c in sorted(counts.items())
                if 0 <= i < len(class_names) and 0 <= j < len(class_names)
            ]
            self.log_table(f"{key}_table", ["true", "pred", "count"], rows)

    def log_artifact_files(
        self,
        *,
        name: str,
        paths: Sequence[Path],
        artifact_type: str = "evaluation",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        import wandb

        existing = [Path(p) for p in paths if Path(p).is_file()]
        if not existing:
            return
        try:
            artifact = wandb.Artifact(
                name=name, type=artifact_type, metadata=metadata or {}
            )
            for path in existing:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as exc:  # noqa: BLE001 — artifacts are best-effort
            logger.warning("Could not log WandB artifact %s: %s", name, exc)

    def finish(self, exit_code: int = 0) -> None:
        if not self.active:
            return
        try:
            self._run.finish(exit_code=exit_code)
        finally:
            self._run = None


@contextmanager
def start_run(
    *,
    name: str,
    job_type: str,
    config: dict[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    notes: str | None = None,
    settings: WandbSettings | None = None,
    enabled: bool | None = None,
) -> Iterator[WandbRun]:
    """Context manager that starts and finishes a WandB run (or a no-op)."""
    resolved = settings or load_wandb_settings(enabled=enabled)
    if not resolved.enabled:
        yield WandbRun(None, resolved)
        return

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping experiment tracking")
        yield WandbRun(None, resolved)
        return

    os.environ.setdefault("WANDB_PROJECT", resolved.project)
    if resolved.entity:
        os.environ.setdefault("WANDB_ENTITY", resolved.entity)
    os.environ["WANDB_MODE"] = resolved.mode

    init_kwargs: dict[str, Any] = {
        "project": resolved.project,
        "name": name,
        "job_type": job_type,
        "config": config or {},
        "tags": list(dict.fromkeys([*resolved.tags, *(tags or ())])),
        # Finish any prior run in this process so train→eval chaining is safe.
        "reinit": "finish_previous",
    }
    if resolved.entity:
        init_kwargs["entity"] = resolved.entity
    if notes:
        init_kwargs["notes"] = notes

    run = None
    exit_code = 0
    try:
        run = wandb.init(**init_kwargs)
        wrapper = WandbRun(run, resolved)
        yield wrapper
    except Exception:
        exit_code = 1
        raise
    finally:
        if run is not None:
            try:
                run.finish(exit_code=exit_code)
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb.finish failed: %s", exc)


def add_wandb_cli_flags(parser: Any) -> None:
    """Attach standard ``--wandb`` / ``--no-wandb`` flags to an argparse parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=None,
        help="Force-enable Weights & Biases logging",
    )
    group.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases logging for this run",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override WANDB_PROJECT (default: smol-doc-analyzer)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional WandB run display name",
    )


def settings_from_args(args: Any) -> WandbSettings:
    """Build settings from argparse namespace produced by ``add_wandb_cli_flags``."""
    return load_wandb_settings(
        enabled=getattr(args, "wandb", None),
        project=getattr(args, "wandb_project", None),
    )
