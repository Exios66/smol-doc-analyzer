"""Tests for claim outcome prediction + evaluation metrics."""

from __future__ import annotations

from pathlib import Path

from src.generation.skeleton_sampler import sample_batch
from src.pipeline.eval_outcome import evaluate_outcomes, write_outcome_report
from src.pipeline.orchestrator import analyze_document
from src.pipeline.outcome import (
    OUTCOME_LABELS,
    derive_expected_outcome,
    features_from_skeleton,
    predict_outcome,
)
from src.utils.config import Config
from src.utils.io import read_json


def test_derive_expected_outcome_rules():
    assert (
        derive_expected_outcome(
            {
                "narrative_complexity": "fraud_flagged",
                "estimated_damage": 25000,
                "deductible": 500,
                "injuries_reported": False,
            }
        )
        == "deny"
    )
    assert (
        derive_expected_outcome(
            {
                "narrative_complexity": "ambiguous",
                "estimated_damage": 2000,
                "deductible": 500,
                "injuries_reported": False,
            }
        )
        == "investigate"
    )
    assert (
        derive_expected_outcome(
            {
                "narrative_complexity": "clean",
                "estimated_damage": 400,
                "deductible": 500,
                "injuries_reported": False,
            }
        )
        == "close_without_payment"
    )
    assert (
        derive_expected_outcome(
            {
                "narrative_complexity": "clean",
                "estimated_damage": 3000,
                "deductible": 500,
                "injuries_reported": False,
            }
        )
        == "pay_full"
    )


def test_skeleton_sampler_sets_expected_outcome():
    cfg = Config.load()
    dist = read_json(cfg.profiles_dir / "insurance_distributions.json")
    schema = read_json(cfg.claim_schema_path)
    skeletons = sample_batch(n=12, seed=7, dist=dist, schema=schema)
    for sk in skeletons:
        assert sk["expected_outcome"] in OUTCOME_LABELS
        assert sk["expected_outcome"] == derive_expected_outcome(features_from_skeleton(sk))


def test_predict_outcome_matches_gold_from_perfect_features():
    skeleton = {
        "narrative_complexity": "standard",
        "document_type": "loss_notice",
        "loss_event": {
            "loss_type": "collision",
            "injuries_reported": False,
            "police_report_filed": True,
        },
        "financials": {
            "estimated_damage": 20000,
            "deductible": 1000,
            "reserve_set": 18000,
        },
    }
    gold = derive_expected_outcome(features_from_skeleton(skeleton))
    skeleton["expected_outcome"] = gold
    pred = predict_outcome(
        fields={
            "estimated_damage": 20000,
            "deductible": 1000,
            "reserve_set": 18000,
            "loss_type": "collision",
        },
        document_type="loss_notice",
        narrative_complexity="standard",
        gold_skeleton=skeleton,
    )
    assert pred["expected_outcome"] == gold
    assert pred["correct"] is True


def test_pipeline_includes_predict_outcome_stage():
    cfg = Config.load()
    text = (
        "AUTOMOBILE LOSS NOTICE\n"
        "Claim Number: CLM-OUT-1\n"
        "Date of Loss: 2024-01-15\n"
        "Loss Type: collision\n"
        "Estimated Damage: $3,200.00\n"
        "Deductible: $500.00\n"
        "Reserve Amount: $2,800.00\n"
        "Injuries Reported: No\n"
        "Complexity assessment: clean.\n"
    )
    result = analyze_document(
        text,
        record_id="outcome-stage-test",
        cfg=cfg,
        enable_vision=False,
    )
    names = [s["stage"] for s in result["stages"]]
    assert names == [
        "to_markdown",
        "classify",
        "extract",
        "vision_llm",
        "predict_outcome",
        "summarize",
    ]
    assert result["expected_outcome"] in OUTCOME_LABELS
    assert result["outcome"]["expected_outcome"] == result["expected_outcome"]
    assert "Predicted claim outcome" in (result.get("memo") or "")


def test_outcome_eval_report(tmp_path: Path):
    cfg = Config.load()
    dist = read_json(cfg.profiles_dir / "insurance_distributions.json")
    schema = read_json(cfg.claim_schema_path)
    skeletons = sample_batch(n=8, seed=3, dist=dist, schema=schema)
    rows = []
    for sk in skeletons:
        rows.append(
            {
                "record_id": sk["claim_id"],
                "claim_id": sk["claim_id"],
                "document_type": sk["document_type"],
                "expected_outcome": sk["expected_outcome"],
                "skeleton": sk,
                "narrative_complexity": sk["narrative_complexity"],
                "text": (
                    f"LOSS NOTICE\nClaim Number: {sk['claim_id']}\n"
                    f"Date of Loss: {sk['loss_event']['date_of_loss']}\n"
                    f"Loss Type: {sk['loss_event']['loss_type']}\n"
                    f"Estimated Damage: ${sk['financials']['estimated_damage']}\n"
                    f"Deductible: ${sk['financials']['deductible']}\n"
                    f"Reserve Amount: ${sk['financials']['reserve_set']}\n"
                    f"Injuries Reported: {'Yes' if sk['loss_event']['injuries_reported'] else 'No'}\n"
                    f"Complexity assessment: {sk['narrative_complexity']}.\n"
                ),
            }
        )
    report = evaluate_outcomes(rows, cfg=cfg, enable_vision=False)
    assert report["n"] == 8
    assert report["accuracy"] is not None
    assert 0.0 <= report["accuracy"] <= 1.0
    json_path, md_path = write_outcome_report(report, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    assert "Claim outcome prediction" in md_path.read_text(encoding="utf-8")
