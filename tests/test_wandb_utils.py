"""Tests for Weights & Biases helpers (no network / no real wandb account)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.wandb_utils import (
    WandbRun,
    add_wandb_cli_flags,
    load_wandb_settings,
    settings_from_args,
    start_run,
)


def test_load_wandb_settings_disabled_via_mode(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    settings = load_wandb_settings()
    assert settings.enabled is False
    assert settings.mode == "disabled"


def test_load_wandb_settings_disabled_via_flag(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    settings = load_wandb_settings()
    assert settings.enabled is False


def test_load_wandb_settings_falls_back_offline_without_key(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setattr(
        "src.utils.wandb_utils._wandb_netrc_available", lambda: False
    )
    settings = load_wandb_settings()
    assert settings.enabled is True
    assert settings.mode == "offline"
    assert settings.project == "test-project"


def test_load_wandb_settings_stays_online_with_netrc(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        "src.utils.wandb_utils._wandb_netrc_available", lambda: True
    )
    settings = load_wandb_settings()
    assert settings.enabled is True
    assert settings.mode == "online"


def test_explicit_enabled_false_overrides_env(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    settings = load_wandb_settings(enabled=False)
    assert settings.enabled is False


def test_cli_flags_and_settings_from_args():
    parser = argparse.ArgumentParser()
    add_wandb_cli_flags(parser)
    args = parser.parse_args(["--no-wandb", "--wandb-project", "cli-proj", "--wandb-run-name", "r1"])
    assert args.wandb is False
    assert args.wandb_project == "cli-proj"
    assert args.wandb_run_name == "r1"
    settings = settings_from_args(args)
    assert settings.enabled is False
    assert settings.project == "cli-proj"


def test_noop_run_when_disabled(tmp_path: Path):
    settings = load_wandb_settings(enabled=False, mode="disabled")
    with start_run(
        name="noop",
        job_type="eval",
        config={"x": 1},
        settings=settings,
    ) as wb:
        assert isinstance(wb, WandbRun)
        assert wb.active is False
        assert wb.report_to == []
        wb.log({"a": 1})
        wb.summary({"b": 2})
        wb.log_table("t", ["c"], [[1]])
        wb.log_confusion_matrix(key="cm", y_true=[0], y_pred=[0], class_names=["a"])
        wb.log_artifact_files(name="art", paths=[tmp_path / "missing.json"])


def test_config_includes_wandb_fields(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "from-config")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_MODE", "offline")
    from src.utils.config import Config

    cfg = Config.load()
    assert cfg.wandb_project == "from-config"
    assert cfg.wandb_entity == "team"
    assert cfg.wandb_mode == "offline"
