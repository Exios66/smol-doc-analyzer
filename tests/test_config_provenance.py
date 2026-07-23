from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import Config, REPO_ROOT
from src.utils.provenance import ProvenanceRecord, log_provenance, read_provenance


def test_config_load_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("SKELETON_OUTPUT_DIR", str(tmp_path / "skeletons"))
    cfg = Config.load()
    assert cfg.taxonomy_path == REPO_ROOT / "taxonomy" / "acord_form_categories.yaml"
    assert cfg.claim_schema_path.exists()
    assert cfg.skeleton_output_dir == tmp_path / "skeletons"
    assert cfg.profiles_dir.name == "profiles"
    assert cfg.sample_corpus_db_path.name == "documents.db"
    assert cfg.rvl_cdip_db_path.name == "rvl_cdip.db"
    assert ".venv" in str(cfg.rvl_cdip_root)
    assert (REPO_ROOT / "data" / "schemas" / "medical_bill_skeleton.schema.json").exists()
    assert (REPO_ROOT / "data" / "schemas" / "salvage_document_skeleton.schema.json").exists()


def test_placeholder_secrets_treated_as_unset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-your-key-here")
    monkeypatch.setenv("WANDB_API_KEY", "changeme")
    cfg = Config.load()
    assert cfg.openrouter_api_key == ""
    assert cfg.wandb_api_key == ""


def test_dotenv_loads_secrets(tmp_path, monkeypatch):
    from src.utils import config as config_mod

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=sk-or-v1-test-secret\nWANDB_API_KEY=wandb-test-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # Real process env must not already carry these
    cfg = config_mod.Config.load()
    assert cfg.openrouter_api_key == "sk-or-v1-test-secret"
    assert cfg.wandb_api_key == "wandb-test-secret"


def test_secrets_status_never_leaks_values(monkeypatch):
    from src.utils.config import secrets_status

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-real-looking")
    status = secrets_status()
    assert status["OPENROUTER_API_KEY"] is True
    assert "sk-or" not in str(status)


def test_provenance_roundtrip(tmp_path):
    log_path = tmp_path / "provenance_log.jsonl"
    record = ProvenanceRecord(
        record_id="test-001",
        stage="corpus_ingest",
        source="unit_test",
        prompt_version="v0",
        model=None,
        extra={"n": 1},
    )
    log_provenance(log_path, record)
    rows = read_provenance(log_path)
    assert len(rows) == 1
    assert rows[0]["record_id"] == "test-001"
    assert rows[0]["stage"] == "corpus_ingest"
    assert json.loads(log_path.read_text(encoding="utf-8").strip())["extra"]["n"] == 1


def test_taxonomy_and_schema_align():
    import yaml

    taxonomy = yaml.safe_load(
        (REPO_ROOT / "taxonomy" / "acord_form_categories.yaml").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (REPO_ROOT / "data" / "schemas" / "claim_skeleton.schema.json").read_text(encoding="utf-8")
    )
    taxonomy_labels = {c["label"] for c in taxonomy["categories"]}
    skeleton_types = set(schema["properties"]["document_type"]["enum"])
    # adjuster_memo is a classifier-recognizable type but not a skeleton input type
    assert skeleton_types <= taxonomy_labels
    assert "adjuster_memo" in taxonomy_labels
    assert "adjuster_memo" not in skeleton_types
