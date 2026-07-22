"""Tests for DICIE FastAPI serve path (optional ``.[serve]`` extra)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.docie.serve import _safe_upload_path, _sanitize_upload_record_id
from src.docie.types import (
    ClassificationResult,
    DociePrediction,
    ExtractionResult,
)


def test_sanitize_upload_record_id_strips_traversal():
    assert _sanitize_upload_record_id("../../../tmp/escape") == "escape"
    assert _sanitize_upload_record_id("..") == "upload"
    assert _sanitize_upload_record_id("sal-log-001") == "sal-log-001"
    assert "/" not in _sanitize_upload_record_id("a/b/c.pdf")
    assert "\\" not in _sanitize_upload_record_id("a\\b\\c")


def test_safe_upload_path_stays_under_temp(tmp_path: Path):
    safe_id, path = _safe_upload_path(tmp_path, "../../../tmp/escape", ".pdf")
    assert safe_id == "escape"
    assert path.parent == tmp_path.resolve()
    assert path.name == "escape.pdf"
    assert path.is_relative_to(tmp_path.resolve())

    safe_id2, path2 = _safe_upload_path(tmp_path, "foo/../../evil", ".png")
    assert safe_id2 == "evil"
    assert path2.is_relative_to(tmp_path.resolve())


def test_predict_sanitizes_traversal_record_id(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.docie import serve as serve_mod
    from src.docie.serve import create_app

    seen: dict = {}

    def fake_process(self, **kwargs):
        seen.update(kwargs)
        path = kwargs.get("pdf_path") or kwargs.get("image_path")
        assert path is not None
        assert "docie_upload_" in str(path)
        assert ".." not in Path(path).name
        return DociePrediction(
            record_id=kwargs["record_id"],
            application="salvage_claims",
            classification=ClassificationResult(
                label="log",
                confidence=0.9,
                backend="test",
            ),
            extraction=ExtractionResult(
                fields={"claim_id": ["CLM-1"]},
                fields_flat={"claim_id": "CLM-1"},
                backend="test",
                document_type="log",
                confidence=0.9,
            ),
            needs_human_review=False,
        )

    monkeypatch.setattr(serve_mod.DociePipeline, "process", fake_process)

    app = create_app(application="salvage_claims")
    client = TestClient(app)
    response = client.post(
        "/v1/predict",
        data={
            "application_name": "salvage_claims",
            "record_id": "../../../tmp/escape",
            "response_only": "true",
        },
        files={"file": ("log.png", b"fakepngbytes", "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["record_id"] == "escape"
    assert seen["record_id"] == "escape"
    assert Path(seen["image_path"]).name == "escape.png"
