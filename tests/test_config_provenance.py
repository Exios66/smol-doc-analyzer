from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import Config, REPO_ROOT
from src.utils.provenance import ProvenanceRecord, log_provenance, read_provenance


def test_config_load_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("SKELETON_OUTPUT_DIR", str(tmp_path / "skeletons"))
    cfg = Config.load()
    assert cfg.taxonomy_path == REPO_ROOT / "taxonomy" / "acord_form_categories.yaml"
    assert cfg.claim_schema_path.exists()
    assert cfg.skeleton_output_dir == tmp_path / "skeletons"
    assert cfg.profiles_dir.name == "profiles"


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
