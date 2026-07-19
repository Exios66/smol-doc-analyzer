from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.metrics import (
    REQUIRED_MEMO_FIELDS,
    annotate_records,
    classification_metrics,
    extraction_metrics,
    score_classification,
    score_extraction,
    score_memo_rubric,
    score_results_file,
    summarize,
    write_summary_csv,
)


def _cls(pred: str, truth: str, backend: str = "local", **extra) -> dict:
    return {
        "task": "classification",
        "backend": backend,
        "model_id": "local" if backend == "local" else f"{backend}/model",
        "example_id": f"{pred}-{truth}",
        "prediction": pred,
        "ground_truth": truth,
        "latency_seconds": 0.1,
        "cost_usd": 0.001,
        "error": None,
        **extra,
    }


def _ext(pred: dict, truth: dict, backend: str = "local", **extra) -> dict:
    return {
        "task": "extraction",
        "backend": backend,
        "model_id": "local",
        "example_id": "ext",
        "prediction": pred,
        "ground_truth": truth,
        "latency_seconds": 0.2,
        "cost_usd": 0.002,
        "error": None,
        **extra,
    }


def test_score_classification_accuracy_and_macro_f1():
    records = [
        _cls("loss_notice", "loss_notice"),
        _cls("loss_notice", "loss_notice"),
        _cls("certificate_evidence", "loss_notice"),
        _cls("certificate_evidence", "certificate_evidence"),
    ]
    accuracy, macro_f1 = score_classification(records)
    assert accuracy == pytest.approx(0.75)
    assert 0.0 < macro_f1 <= 1.0


def test_classification_metrics_includes_auc_and_f1_averages():
    y_true = ["log", "log", "sales", "other"]
    y_pred = ["log", "sales", "sales", "other"]
    y_scores = [
        {"log": 0.8, "sales": 0.1, "other": 0.1},
        {"log": 0.2, "sales": 0.7, "other": 0.1},
        {"log": 0.1, "sales": 0.8, "other": 0.1},
        {"log": 0.1, "sales": 0.1, "other": 0.8},
    ]
    metrics = classification_metrics(
        y_true, y_pred, y_scores=y_scores, labels=["log", "sales", "other"]
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert "macro_f1" in metrics and "micro_f1" in metrics and "weighted_f1" in metrics
    assert metrics["auc_ovr"] is not None
    assert metrics["auc_ovo"] is not None
    # log: TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1=2/3
    assert metrics["per_class"]["log"]["precision"] == pytest.approx(1.0)
    assert metrics["per_class"]["log"]["recall"] == pytest.approx(0.5)
    assert metrics["per_class"]["log"]["f1"] == pytest.approx(2 / 3)


def test_extraction_metrics_per_field_f1_harmonic_mean():
    records = [
        _ext(
            {"claim_id": "CLM-1", "vin": "1HGCM82633A004352"},
            {"claim_id": "CLM-1", "vin": "1HGCM82633A004352", "year": "2018"},
        ),
        _ext(
            {"claim_id": "WRONG", "vin": "1HGCM82633A004352"},
            {"claim_id": "CLM-2", "vin": "1HGCM82633A004352"},
        ),
    ]
    metrics = extraction_metrics(
        records, fuzzy_fields=set(), fields=["claim_id", "vin", "year"]
    )
    assert "micro_f1" in metrics and "macro_f1" in metrics
    assert "claim_id" in metrics["per_field"]
    # VIN exact on both → F1 1.0
    assert metrics["per_field"]["vin"]["f1"] == pytest.approx(1.0)
    # year missing on both preds, gold once → recall 0
    assert metrics["per_field"]["year"]["recall"] == pytest.approx(0.0)
    # Harmonic mean identity
    p = metrics["per_field"]["vin"]["precision"]
    r = metrics["per_field"]["vin"]["recall"]
    assert metrics["per_field"]["vin"]["f1"] == pytest.approx(2 * p * r / (p + r))


def test_score_extraction_exact_and_fuzzy():
    records = [
        _ext(
            {"claim_id": "CLM-1", "location": "Madison,  WI"},
            {"claim_id": "CLM-1", "location": "madison, wi"},
        )
    ]
    # Exact match fails on location whitespace/case.
    exact = score_extraction(records, fuzzy_fields=set())
    assert exact < 1.0
    fuzzy = score_extraction(records, fuzzy_fields={"location"})
    assert fuzzy == pytest.approx(1.0)


def test_score_extraction_mismatch_counts_fp_and_fn():
    # Wrong value for a present ground-truth field must not report recall=1.0.
    records = [_ext({"claim_id": "WRONG"}, {"claim_id": "CLM-1"})]
    f1 = score_extraction(records, fuzzy_fields=set())
    assert f1 == pytest.approx(0.0)


def test_score_classification_normalizes_label_format():
    records = [
        _cls("Loss Notice", "loss_notice"),
        _cls("certificate-evidence", "certificate_evidence"),
    ]
    accuracy, _ = score_classification(records)
    assert accuracy == pytest.approx(1.0)


def test_parse_error_treated_as_error_in_annotate():
    rows = [
        {
            "task": "extraction",
            "backend": "openai",
            "model_id": "openai/gpt-4o",
            "prediction": {"_parse_error": True, "_raw": "nope"},
            "ground_truth": {"claim_id": "CLM-1"},
            "latency_seconds": 0.1,
            "cost_usd": 0.01,
            "error": None,
        }
    ]
    annotated = annotate_records(rows)
    assert annotated[0]["error"] == "prediction_parse_error"
    assert annotated[0]["correct"] is False
    assert annotated[0]["score"] == 0.0


def test_score_extraction_handles_list_values_and_parse_errors():
    records = [
        _ext(
            {"claim_id": ["CLM-1"], "_parse_error": True, "_raw": "{bad"},
            {"claim_id": "CLM-1", "state": "WI"},
        )
    ]
    # claim_id matches after list flatten; state is a false negative.
    f1 = score_extraction(records, fuzzy_fields=set())
    assert 0.0 < f1 < 1.0


def test_score_memo_rubric_keyword_proxy():
    empty = score_memo_rubric("")
    assert empty == 0.0
    memo = (
        "ADJUSTER MEMO — CLM-2026-000101\n"
        "Summary\n"
        "Analysis of whether coverage appears supported.\n"
        "Next Steps\n"
        "- Confirm coverage grant/denial points in writing\n"
    )
    score = score_memo_rubric(memo)
    assert score >= 0.75
    assert len(REQUIRED_MEMO_FIELDS) == 4


def test_annotate_and_summarize(tmp_path: Path):
    rows = [
        _cls("loss_notice", "loss_notice", backend="local"),
        _cls("certificate_evidence", "loss_notice", backend="local"),
        _ext(
            {"claim_id": "CLM-1", "state": "WI"},
            {"claim_id": "CLM-1", "state": "WI"},
            backend="openai",
        ),
        {
            "task": "memo_generation",
            "backend": "anthropic",
            "model_id": "anthropic/claude-sonnet-4.5",
            "example_id": "memo-1",
            "prediction": "ADJUSTER MEMO CLM-1\nCoverage determination pending.\nNext Steps\nAdjuster notes follow.",
            "ground_truth": "reference",
            "latency_seconds": 1.0,
            "cost_usd": 0.01,
            "error": None,
            "judge_score": 4.0,
        },
        {
            "task": "classification",
            "backend": "local",
            "model_id": "local",
            "example_id": "err",
            "prediction": None,
            "ground_truth": "loss_notice",
            "latency_seconds": 0.0,
            "cost_usd": 0.0,
            "error": "boom",
        },
    ]
    path = tmp_path / "eval_results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summaries = summarize(path)
    by_key = {(s.task, s.backend): s for s in summaries}

    cls_local = by_key[("classification", "local")]
    assert cls_local.n_examples == 3
    assert cls_local.error_rate == pytest.approx(1 / 3)
    assert cls_local.accuracy == pytest.approx(0.5)  # 1/2 clean
    assert cls_local.macro_f1 is not None

    ext = by_key[("extraction", "openai")]
    assert ext.macro_f1 == pytest.approx(1.0)
    assert ext.avg_cost_per_doc_usd == pytest.approx(0.002)

    memo = by_key[("memo_generation", "anthropic")]
    assert memo.rubric_coverage is not None and memo.rubric_coverage > 0.5
    assert memo.judge_score_avg == pytest.approx(4.0)

    annotated = annotate_records([dict(r) for r in rows])
    assert annotated[0]["correct"] is True
    assert annotated[1]["correct"] is False
    assert annotated[2]["score"] == pytest.approx(1.0)


def test_score_results_file_cli_outputs(tmp_path: Path):
    rows = [
        _cls("loss_notice", "loss_notice"),
        _ext({"claim_id": "A"}, {"claim_id": "A"}),
    ]
    results = tmp_path / "eval_results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out_csv = tmp_path / "summary.csv"
    out_json = tmp_path / "summary.json"

    summaries = score_results_file(results, out_csv, output_json=out_json)
    assert out_csv.exists() and out_json.exists()
    assert len(summaries) == 2

    # Annotations written back in place.
    annotated = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert all("correct" in r and "score" in r for r in annotated)

    write_summary_csv([], tmp_path / "empty.csv")
    assert (tmp_path / "empty.csv").read_text(encoding="utf-8") == ""


def test_metrics_main(tmp_path: Path):
    from evaluation.metrics import main

    rows = [_cls("loss_notice", "loss_notice")]
    results = tmp_path / "eval_results.jsonl"
    results.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    out = tmp_path / "summary.csv"
    main(["--results", str(results), "--output", str(out)])
    assert out.exists()
    assert out.with_suffix(".json").exists()
