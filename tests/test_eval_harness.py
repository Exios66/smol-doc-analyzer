from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.cost_model_helpers import frontier_cost_usd, local_cost_per_call, local_cost_per_doc_at_throughput
from evaluation.eval_harness import (
    EvalResult,
    ModelPricing,
    compute_cost,
    load_eval_set,
    load_pricing,
    parse_prediction,
    run_eval,
    write_outputs,
)
from src.utils.config import REPO_ROOT
from src.utils.prompts import load_prompt
from src.utils.provenance import ProvenanceLogger


def test_load_pricing():
    frontier, gpu_rate = load_pricing()
    assert "anthropic" in frontier and "openai" in frontier
    assert frontier["anthropic"].input_per_million > 0
    assert gpu_rate > 0


def test_compute_cost_and_helpers():
    pricing = ModelPricing(input_per_million=3.0, output_per_million=15.0)
    assert compute_cost(pricing, 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert frontier_cost_usd(3.0, 15.0, 500_000, 100_000) == pytest.approx(3.0)
    assert local_cost_per_call(0.80, 3.6) == pytest.approx(0.0008)
    assert local_cost_per_doc_at_throughput(0.80, 800) == pytest.approx(0.001)


def test_parse_prediction_variants():
    assert parse_prediction("classification", " Loss_Notice \n") == "loss_notice"
    assert parse_prediction("extraction", '{"claim_id": "CLM-1"}') == {"claim_id": "CLM-1"}
    assert parse_prediction("extraction", 'Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    bad = parse_prediction("extraction", "not json")
    assert bad["_parse_error"] is True
    assert parse_prediction("memo_generation", "  memo text  ") == "memo text"


def test_load_prompt_eval_templates():
    clf = load_prompt("eval_classification")
    assert "{document_text}" in clf
    assert "loss_notice" in clf
    ext = load_prompt("eval_extraction")
    assert "JSON" in ext
    memo = load_prompt("eval_memo_generation")
    assert "{extracted_fields}" in memo


def test_load_eval_set_caps_per_task():
    path = REPO_ROOT / "data" / "eval" / "eval_set.jsonl"
    rows = load_eval_set(path, n_samples=1)
    tasks = [r["task"] for r in rows]
    assert tasks.count("classification") == 1
    assert "extraction" in tasks
    assert "memo_generation" in tasks


def test_load_eval_set_zero_samples_means_empty():
    path = REPO_ROOT / "data" / "eval" / "eval_set.jsonl"
    assert load_eval_set(path, n_samples=0) == []


def test_load_eval_set_rejects_negative():
    path = REPO_ROOT / "data" / "eval" / "eval_set.jsonl"
    with pytest.raises(ValueError, match="n-samples"):
        load_eval_set(path, n_samples=-1)


def test_load_pricing_creates_missing_file(tmp_path: Path):
    from evaluation.eval_harness import load_pricing

    path = tmp_path / "pricing.yaml"
    frontier, rate = load_pricing(path)
    assert path.exists()
    assert "anthropic" in frontier and rate > 0


def test_dry_run_prints_plan(capsys):
    examples = [
        {
            "example_id": "cls-x",
            "task": "classification",
            "prompt_fields": {"document_text": "x"},
            "ground_truth": "loss_notice",
        }
    ]
    results = run_eval(
        tasks=["classification"],
        backends={},  # unused in dry-run path once we pass empty — still iterates backends
        eval_set=examples,
        frontier_pricing={},
        run_id="testrun1",
        dry_run=True,
    )
    # With empty backends, dry-run still yields no results but shouldn't crash.
    assert results == []

    class Dummy:
        model_slug = "dummy"

        def run(self, task, example):  # pragma: no cover - not called in dry-run
            raise AssertionError("should not run")

    results = run_eval(
        tasks=["classification"],
        backends={"local": Dummy()},  # type: ignore[arg-type]
        eval_set=examples,
        frontier_pricing={},
        run_id="testrun2",
        dry_run=True,
    )
    assert results == []
    out = capsys.readouterr().out
    assert "[dry-run] would call local for classification/cls-x" in out


def test_write_outputs_jsonl_csv_and_provenance(tmp_path):
    results = [
        EvalResult(
            run_id="abc12345",
            task="classification",
            backend="local",
            model_id="local",
            example_id="cls-001",
            prediction="loss_notice",
            ground_truth="loss_notice",
            input_tokens=0,
            output_tokens=0,
            latency_seconds=0.01,
            cost_usd=0.000002,
        ),
        EvalResult(
            run_id="abc12345",
            task="extraction",
            backend="openai",
            model_id="openai/gpt-4o",
            example_id="ext-001",
            prediction={"claim_id": "CLM-1"},
            ground_truth={"claim_id": "CLM-1"},
            input_tokens=100,
            output_tokens=50,
            latency_seconds=0.5,
            cost_usd=0.001,
        ),
    ]
    prov_path = tmp_path / "provenance_log.jsonl"
    provenance = ProvenanceLogger(run_id="abc12345", tag="eval_comparison", log_path=prov_path)
    out_dir = tmp_path / "results"
    write_outputs(results, out_dir, provenance)

    jsonl_rows = [
        json.loads(line)
        for line in (out_dir / "eval_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(jsonl_rows) == 2
    assert jsonl_rows[0]["run_id"] == "abc12345"
    assert jsonl_rows[0]["provenance_tag"] == "eval_comparison"
    assert "timestamp" in jsonl_rows[0]

    csv_text = (out_dir / "eval_results.csv").read_text(encoding="utf-8")
    assert "classification" in csv_text
    assert "claim_id" in csv_text

    mirrored = [
        json.loads(line) for line in prov_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(mirrored) == 2
    assert mirrored[0]["stage"] == "eval_comparison"


def test_cli_dry_run(capsys):
    from evaluation.eval_harness import main

    eval_set = REPO_ROOT / "data" / "eval" / "eval_set.jsonl"
    main(
        [
            "--eval-set",
            str(eval_set),
            "--tasks",
            "classification",
            "extraction",
            "--backends",
            "local",
            "anthropic",
            "--n-samples",
            "1",
            "--output-dir",
            str(Path("/tmp/unused")),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "would call local for classification/" in out
    assert "would call anthropic for extraction/" in out
