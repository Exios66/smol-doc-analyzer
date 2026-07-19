"""Tests for TF-IDF + Random Forest classification helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.classification.random_forest import (
    SURFACE_HANDWRITING_OCR,
    SURFACE_TYPED,
    assign_split_column,
    build_document_type_pipeline,
    evaluate_classifier,
    load_text_handwriting_corpus,
    log_random_forest_to_wandb,
    save_random_forest_bundle,
    top_tfidf_feature_importances,
    write_predictions_jsonl,
)
from src.utils.config import Config
from src.utils.io import write_jsonl
from src.utils.wandb_utils import load_wandb_settings


def _write_mini_corpus(tmp: Path) -> tuple[Path, Path]:
    docs = [
        {
            "record_id": "r1",
            "claim_id": "c1",
            "document_type": "loss_notice",
            "text": "PROPERTY LOSS NOTICE Claim Number CLM-1 Date of Loss water damage",
        },
        {
            "record_id": "r2",
            "claim_id": "c2",
            "document_type": "repair_estimate",
            "text": "REPAIR ESTIMATE Statement of charges body shop labor parts total",
        },
        {
            "record_id": "r3",
            "claim_id": "c3",
            "document_type": "loss_notice",
            "text": "AUTOMOBILE LOSS NOTICE Claim Number CLM-3 collision report filed",
        },
        {
            "record_id": "r4",
            "claim_id": "c4",
            "document_type": "repair_estimate",
            "text": "REPAIR ESTIMATE Estimator total replacement cost materials",
        },
        {
            "record_id": "r5",
            "claim_id": "c5",
            "document_type": "claims_correspondence",
            "text": "Dear Claimant claim status update sincerely claims department",
        },
        {
            "record_id": "r6",
            "claim_id": "c6",
            "document_type": "claims_correspondence",
            "text": "Re Claim status letter to claimant next steps sincerely",
        },
    ]
    noisy = [{**d, "text": d["text"].replace("e", "3").replace("a", "@"), "is_noisy": True} for d in docs]
    docs_path = tmp / "documents_from_skeletons_n6_seed0.jsonl"
    noisy_path = tmp / "noisy_from_documents_from_skeletons_n6_seed0.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(noisy_path, noisy)
    return docs_path, noisy_path


def test_load_and_train_random_forest(tmp_path: Path):
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path)
    assert len(frame) == 12
    assert set(frame["surface"]) == {SURFACE_TYPED, SURFACE_HANDWRITING_OCR}

    frame = assign_split_column(frame, splits_path=tmp_path / "missing_splits.json")
    assert set(frame["split"]) <= {"train", "val", "test"}

    # Tiny vocab: lower min_df via direct pipeline params
    model = build_document_type_pipeline(n_estimators=20, max_features=500, random_state=0)
    model.named_steps["tfidf"].set_params(min_df=1)
    model.fit(frame["text"], frame["document_type"])
    metrics = evaluate_classifier(model, frame["text"], frame["document_type"])
    assert metrics["accuracy"] >= 0.5
    assert len(metrics["predictions"]) == len(frame)

    fi = top_tfidf_feature_importances(model, top_k=5)
    assert isinstance(fi, pd.DataFrame)
    assert len(fi) <= 5

    out = tmp_path / "rf_out"
    save_random_forest_bundle(model, out, metrics, meta={"smoke": True})
    assert (out / "random_forest_pipeline.joblib").exists()
    assert (out / "eval_metrics.json").exists()

    write_predictions_jsonl(
        out / "preds.jsonl",
        record_ids=frame["record_id"].tolist(),
        y_true=frame["document_type"].tolist(),
        y_pred=metrics["predictions"],
        surfaces=frame["surface"].tolist(),
        confidences=metrics["max_proba"],
    )
    assert (out / "preds.jsonl").exists()


def test_log_random_forest_to_wandb_noop_when_disabled(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path)
    model = build_document_type_pipeline(n_estimators=20, max_features=500, random_state=0)
    model.named_steps["tfidf"].set_params(min_df=1)
    model.fit(frame["text"], frame["document_type"])
    metrics = evaluate_classifier(model, frame["text"], frame["document_type"])
    log_random_forest_to_wandb(
        doc_metrics=metrics,
        surface_metrics=None,
        config={"smoke": True},
        y_true=frame["document_type"].tolist(),
        wandb_settings=load_wandb_settings(enabled=False),
        run_name="rf-test-disabled",
    )


def test_notebook_exists():
    nb = Path("notebooks/random_forest_text_handwriting_classification.ipynb")
    assert nb.exists()
    text = nb.read_text(encoding="utf-8")
    assert "RandomForest" in text or "random_forest" in text
    assert "handwriting" in text.lower()
    assert "log_random_forest_to_wandb" in text
