"""Paper-aligned DICIE evaluation (Table I / Table II)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.docie.eval import (
    DEFAULT_EVAL_SET,
    evaluate_application,
    render_markdown_report,
    write_reports,
)
from src.utils.config import Config

REPO = Path(__file__).resolve().parents[1]


def test_docie_eval_set_exists_and_balanced():
    assert DEFAULT_EVAL_SET.exists()
    rows = [
        line
        for line in DEFAULT_EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 20
    import json

    parsed = [json.loads(r) for r in rows]
    apps = {r["application"] for r in parsed}
    assert "medical_bills" in apps and "salvage_claims" in apps
    for r in parsed:
        assert "document_type" in r
        assert "ground_truth_fields" in r


@pytest.mark.parametrize("application", ["medical_bills", "salvage_claims"])
def test_docie_eval_paper_thresholds(application: str, tmp_path: Path):
    cfg = Config.load()
    payload = evaluate_application(
        application,
        eval_path=DEFAULT_EVAL_SET,
        cfg=cfg,
        run_ocr=False,
    )
    clf = payload["classification"]
    ext = payload["extraction"]

    assert clf["n"] >= 8
    assert clf["accuracy"] >= 0.90
    assert clf["auc_ovr"] is not None
    assert clf["auc_ovo"] is not None
    assert ext["micro_f1"] >= 0.85

    # Majority of paper fields with support should clear 0.85 F1
    supported = [
        (f, s)
        for f, s in ext["per_field"].items()
        if s.get("support", 0) > 0
    ]
    assert supported
    n_pass = sum(1 for _, s in supported if s["f1"] >= 0.85)
    assert n_pass >= max(1, int(0.5 * len(supported)))

    json_path, md_path = write_reports(payload, reports_dir=tmp_path)
    assert json_path.exists() and md_path.exists()
    md = render_markdown_report(payload)
    assert "Table I" in md and "Table II" in md
    assert "Accuracy" in md and "Micro F1" in md
