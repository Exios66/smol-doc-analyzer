"""Unit tests for Braintrust RVL fixed-size experiment helpers (no live APIs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.braintrust_eval import classifier as clf
from src.braintrust_eval.cost_estimate import ModelRates, project_costs, project_scale_table
from src.braintrust_eval.dataset import (
    build_fixed_size_sampled,
    load_samples,
    pad_to_square_png,
)
from src.braintrust_eval.eval_runner import exact_match_scorer, run_local_loop
from src.braintrust_eval.prompt_improve import format_error_block, select_errors
from src.rvl_cdip.paths import LABEL_NAMES
from src.utils.llm_client import _extract_message_fields, _extract_usage


def test_pad_to_square_png(tmp_path: Path):
    pytest.importorskip("PIL")
    from PIL import Image

    src = tmp_path / "src.png"
    Image.new("RGB", (200, 100), color=(10, 20, 30)).save(src)
    dest = tmp_path / "out.png"
    geom = pad_to_square_png(src, dest, size=256)
    assert dest.is_file()
    with Image.open(dest) as im:
        assert im.size == (256, 256)
    assert geom["out_w"] == 256


def test_build_placeholder_dataset(tmp_path: Path):
    pytest.importorskip("PIL")
    out = tmp_path / "fixed"
    manifest = build_fixed_size_sampled(
        out_dir=out,
        n_per_class=2,
        seed=7,
        size=128,
        placeholder=True,
    )
    assert manifest["total_samples"] == 2 * 16
    samples = load_samples(out / "samples.jsonl")
    assert len(samples) == 32
    assert all(Path(s["fixed_image_path"]).is_file() for s in samples)
    assert set(s["label"] for s in samples) == set(LABEL_NAMES)


def test_classify_dry_run(tmp_path: Path):
    pytest.importorskip("PIL")
    from PIL import Image

    img = tmp_path / "page.png"
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(img)
    result = clf.classify_image(
        image_path=img,
        expected_label="invoice",
        document_id="doc-1",
        dry_run=True,
    )
    assert result.exact_match
    assert result.prediction == "invoice"
    assert result.dry_run is True


def test_exact_match_scorer():
    assert exact_match_scorer({}, {"prediction": "Invoice"}, "invoice") == 1.0
    assert exact_match_scorer({}, {"prediction": "memo"}, "letter") == 0.0
    assert exact_match_scorer({}, "form", "form") == 1.0
    # Spaced gold vs underscore prediction (capstone / RVL bridge)
    assert (
        exact_match_scorer({}, {"prediction": "file_folder"}, "file folder") == 1.0
    )
    assert (
        exact_match_scorer({}, {"prediction": "news article"}, "news_article") == 1.0
    )


def test_capstone_prompt_and_clean_prediction():
    prompt = clf.render_classification_prompt()
    assert "file_folder" in prompt
    assert "scientific_publication" in prompt
    assert "Output only the class name" in prompt
    # Purpose-first boundary rules for weak / ambiguous classes
    assert "primary purpose" in prompt
    assert "Tie-breaker" in prompt
    assert "MSDS" in prompt
    assert "prefer budget over form" in prompt
    assert "prefer presentation over memo" in prompt
    assert "prefer specification over form" in prompt
    assert "prefer questionnaire over form" in prompt
    assert "Classification policy" in prompt
    assert "Classification policy" in clf.CLASSIFICATION_PROMPT
    assert clf.clean_prediction("I think this is a news_article page.") == "news_article"
    assert clf.normalize_capstone_label("file folder") == "file_folder"
    assert clf.to_underscore_label("scientific report") == "scientific_report"


def test_run_local_loop_dry(tmp_path: Path):
    pytest.importorskip("PIL")
    out = tmp_path / "fixed"
    build_fixed_size_sampled(out_dir=out, n_per_class=1, size=64, placeholder=True)
    # Point load_samples via run_local_loop samples arg
    samples = load_samples(out / "samples.jsonl")
    summary = run_local_loop(
        samples=samples,
        dry_run=True,
        out_dir=tmp_path / "runs",
    )
    assert summary["n"] == 16
    assert summary["accuracy_exact_match"] == 1.0
    assert Path(summary["predictions_path"]).is_file()


def test_cost_projection_kimi():
    row = project_costs(model_id="moonshotai/kimi-k3", n_images=160)
    assert row["n_images"] == 160
    assert row["estimated_cost_usd"] > 0
    # Spot-check against the published ~$0.48-ish ballpark for 160 imgs
    # (user table scaled from 800@$2.48 → 160 ≈ $0.50)
    assert 0.1 < row["estimated_cost_usd"] < 2.0

    table = project_scale_table(
        model_ids=["moonshotai/kimi-k3", "anthropic/claude-sonnet-4.5"],
        counts=[160, 800],
    )
    assert len(table) == 4


def test_model_rates_cached_discount():
    rates = ModelRates(
        input_per_million=1.0,
        output_per_million=2.0,
        cached_input_per_million=0.1,
    )
    # 1M prompt of which 500k cached + 100k completion
    cost = rates.cost_usd(
        prompt_tokens=1_000_000,
        completion_tokens=100_000,
        prompt_cached_tokens=500_000,
    )
    assert abs(cost - (0.5 * 1.0 + 0.5 * 0.1 + 0.1 * 2.0)) < 1e-9


def test_select_errors_prefers_reasoning(tmp_path: Path):
    preds = [
        {"label": "memo", "prediction": "letter", "document_id": "a", "reasoning": None},
        {
            "label": "form",
            "prediction": "invoice",
            "document_id": "b",
            "reasoning": "looks like tables",
        },
        {"label": "email", "prediction": "email", "document_id": "c"},
    ]
    errs = select_errors(preds, limit=10)
    assert len(errs) == 2
    assert errs[0]["document_id"] == "b"
    block = format_error_block(errs)
    assert "looks like tables" in block


def test_extract_reasoning_and_usage():
    class Msg:
        content = "invoice"
        reasoning = "I see a tabular layout with amounts."
        reasoning_content = None
        reasoning_details = None

    content, reasoning = _extract_message_fields(Msg())
    assert content == "invoice"
    assert "tabular" in (reasoning or "")

    usage = _extract_usage(
        type(
            "R",
            (),
            {
                "usage": type(
                    "U",
                    (),
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                        "prompt_tokens_details": type(
                            "P", (), {"cached_tokens": 40}
                        )(),
                        "completion_tokens_details": type(
                            "C", (), {"reasoning_tokens": 30}
                        )(),
                    },
                )()
            },
        )()
    )
    assert usage["prompt_cached_tokens"] == 40
    assert usage["completion_reasoning_tokens"] == 30


def test_default_reasoning_config():
    assert clf.default_reasoning_config("moonshotai/kimi-k3")["exclude"] is False
    assert clf.default_reasoning_config("deepseek/deepseek-r1")["effort"] == "medium"
    assert clf.default_reasoning_config("openai/gpt-4o-mini") is None


def test_cli_status(monkeypatch, capsys):
    from src.braintrust_eval.__main__ import main

    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "braintrust_project" in payload
    assert payload["braintrust_project"] == "DSHB_amfam_capstone_2026"


def test_config_braintrust_fields(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-test-key")
    monkeypatch.setenv("BRAINTRUST_PROJECT", "MyProj")
    from src.utils.config import Config

    cfg = Config.load()
    assert cfg.braintrust_api_key == "bt-test-key"
    assert cfg.braintrust_project == "MyProj"
    assert cfg.braintrust_dataset == "fixed_size_sampled"
