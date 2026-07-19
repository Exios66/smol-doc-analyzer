"""Tests for TF-IDF + Random Forest classification helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.classification.random_forest import (
    SURFACE_HANDWRITING_OCR,
    SURFACE_TYPED,
    _largest_corpus_path,
    assign_split_column,
    build_document_type_pipeline,
    confidence_diagnostics,
    corpus_n_from_path,
    ensure_seed_corpus,
    evaluate_by_surface,
    evaluate_classifier,
    fit_pipeline_with_tree_curve,
    load_text_handwriting_corpus,
    log_random_forest_to_wandb,
    pipeline_from_preset,
    profile_corpus,
    save_random_forest_bundle,
    top_confusion_pairs,
    top_tfidf_feature_importances,
    write_predictions_jsonl,
)
from src.classification.train_random_forest import run_capacity_sweep, train
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
        {
            "record_id": "r7",
            "claim_id": "c7",
            "document_type": "loss_notice",
            "text": "HOMEOWNERS LOSS NOTICE Claim Number CLM-7 fire damage kitchen",
        },
        {
            "record_id": "r8",
            "claim_id": "c8",
            "document_type": "repair_estimate",
            "text": "REPAIR ESTIMATE labor hours paint materials subtotal grand total",
        },
    ]
    noisy = [
        {**d, "text": d["text"].replace("e", "3").replace("a", "@"), "is_noisy": True}
        for d in docs
    ]
    docs_path = tmp / "documents_from_skeletons_n8_seed0.jsonl"
    noisy_path = tmp / "noisy_from_documents_from_skeletons_n8_seed0.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(noisy_path, noisy)
    return docs_path, noisy_path


def test_load_and_train_random_forest(tmp_path: Path):
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path)
    assert len(frame) == 16
    assert set(frame["surface"]) == {SURFACE_TYPED, SURFACE_HANDWRITING_OCR}

    frame = assign_split_column(frame, splits_path=tmp_path / "missing_splits.json")
    assert set(frame["split"]) <= {"train", "val", "test"}

    model = build_document_type_pipeline(n_estimators=20, max_features=500, random_state=0)
    from src.classification.random_forest import set_vectorizer_min_df

    set_vectorizer_min_df(model, 1)
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


def test_profile_slice_confidence_helpers(tmp_path: Path):
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path)
    frame = assign_split_column(frame, splits_path=tmp_path / "missing_splits.json")
    profile = profile_corpus(frame)
    assert profile["n_rows"] == 16
    assert profile["n_typed"] == 8
    assert profile["n_ocr"] == 8
    assert profile["n_classes"] == 3

    model, _ = pipeline_from_preset("shallow", smoke=True)
    from src.classification.random_forest import set_vectorizer_min_df

    set_vectorizer_min_df(model, 1)
    model.fit(frame["text"], frame["document_type"])
    metrics = evaluate_classifier(model, frame["text"], frame["document_type"])

    slices = evaluate_by_surface(model, frame)
    assert SURFACE_TYPED in slices
    assert SURFACE_HANDWRITING_OCR in slices
    assert "accuracy" in slices[SURFACE_TYPED]
    assert "macro_f1" in slices[SURFACE_HANDWRITING_OCR]

    conf = confidence_diagnostics(
        frame["document_type"].tolist(),
        metrics["predictions"],
        metrics["max_proba"],
        n_bins=5,
    )
    assert "ece" in conf
    assert "mean_confidence" in conf
    assert len(conf["bins"]) == 5

    pairs = top_confusion_pairs(metrics, top_k=5)
    assert isinstance(pairs, list)


def test_fit_pipeline_with_tree_curve(tmp_path: Path):
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path)
    model, _ = pipeline_from_preset("shallow", smoke=True)
    from src.classification.random_forest import set_vectorizer_min_df

    set_vectorizer_min_df(model, 1)
    history = fit_pipeline_with_tree_curve(
        model,
        frame["text"],
        frame["document_type"],
        val_texts=frame["text"],
        val_y=frame["document_type"],
        tree_chunk=5,
    )
    assert len(history) >= 2
    assert history[-1]["n_estimators"] == model.named_steps["rf"].n_estimators
    assert "val_macro_f1" in history[-1]


def test_capacity_sweep_selects_preset(tmp_path: Path):
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path)
    frame = assign_split_column(frame, splits_path=tmp_path / "missing_splits.json")
    train_df = frame[frame["split"] == "train"]
    val_df = frame[frame["split"] == "val"]
    if val_df.empty:
        val_df = train_df
    best, rows = run_capacity_sweep(
        train_df,
        val_df,
        ["shallow", "balanced"],
        smoke=True,
    )
    assert best in {"shallow", "balanced"}
    assert len(rows) == 2
    assert all("val_macro_f1" in r for r in rows)


def test_multilayer_train_writes_sweep(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    out = tmp_path / "rf_multi"
    result = train(
        docs_path=docs_path,
        noisy_path=noisy_path,
        out_dir=out,
        ensure_data=False,
        presets=["shallow", "char_robust"],
        smoke=True,
        wandb_settings=load_wandb_settings(enabled=False),
        wandb_run_name="rf-test-multi",
    )
    assert result == out
    assert (out / "sweep_results.json").exists()
    assert (out / "layer_diagnostics.json").exists()
    assert (out / "random_forest_pipeline.joblib").exists()
    assert (out / "eval_metrics.json").exists()
    import json

    diag = json.loads((out / "layer_diagnostics.json").read_text(encoding="utf-8"))
    assert "tree_history" in diag
    assert len(diag["tree_history"]) >= 2


def test_log_random_forest_to_wandb_multilayer_payload(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    docs_path, noisy_path = _write_mini_corpus(tmp_path)
    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path)
    model, _ = pipeline_from_preset("shallow", smoke=True)
    from src.classification.random_forest import set_vectorizer_min_df

    set_vectorizer_min_df(model, 1)
    model.fit(frame["text"], frame["document_type"])
    metrics = evaluate_classifier(model, frame["text"], frame["document_type"])
    log_random_forest_to_wandb(
        doc_metrics=metrics,
        surface_metrics=None,
        config={"smoke": True},
        y_true=frame["document_type"].tolist(),
        wandb_settings=load_wandb_settings(enabled=False),
        run_name="rf-test-disabled",
        layer_payload={
            "data_profile": profile_corpus(frame),
            "sweep_rows": [
                {
                    "preset": "shallow",
                    "val_accuracy": 0.9,
                    "val_macro_f1": 0.8,
                    "val_weighted_f1": 0.85,
                    "n_estimators": 20,
                    "analyzer": "word",
                    "max_depth": 12,
                }
            ],
            "best_preset": "shallow",
            "best_val_macro_f1": 0.8,
            "slice_metrics": evaluate_by_surface(model, frame),
            "confidence": confidence_diagnostics(
                frame["document_type"].tolist(),
                metrics["predictions"],
                metrics.get("max_proba") or [],
            ),
            "top_features": top_tfidf_feature_importances(model, top_k=3).to_dict(
                orient="records"
            ),
            "confusion_pairs": top_confusion_pairs(metrics),
        },
    )


def test_log_random_forest_emits_many_steps(monkeypatch):
    """WandB charts need many ``_step`` values — not one bulk log."""
    from unittest.mock import MagicMock, patch

    from src.utils import wandb_utils

    fake_run = MagicMock()
    logs: list[tuple[int | None, dict]] = []

    def fake_log(data, step=None):
        logs.append((step, dict(data)))

    fake_run.log.side_effect = fake_log
    fake_run.summary = {}
    settings = load_wandb_settings(enabled=True, mode="offline")

    with (
        patch("wandb.init", return_value=fake_run),
        patch("wandb.Table", MagicMock()),
        patch("wandb.plot.confusion_matrix", MagicMock(return_value="cm")),
        patch("wandb.Artifact", MagicMock()),
        patch.object(wandb_utils, "load_wandb_settings", return_value=settings),
    ):
        log_random_forest_to_wandb(
            doc_metrics={
                "accuracy": 0.9,
                "macro_f1": 0.8,
                "weighted_f1": 0.85,
                "n": 10,
                "labels": ["a", "b"],
                "predictions": ["a", "b"] * 5,
                "classification_report": {
                    "a": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 5},
                    "b": {"precision": 0.8, "recall": 0.8, "f1-score": 0.8, "support": 5},
                },
                "confusion_matrix": [[5, 0], [0, 5]],
            },
            y_true=["a", "b"] * 5,
            wandb_settings=settings,
            run_name="unit-many-steps",
            layer_payload={
                "data_profile": {
                    "n_rows": 10,
                    "n_classes": 2,
                    "n_typed": 5,
                    "n_ocr": 5,
                    "typed_frac": 0.5,
                    "ocr_frac": 0.5,
                    "class_counts": {"a": 5, "b": 5},
                },
                "sweep_rows": [
                    {
                        "preset": "shallow",
                        "val_accuracy": 0.8,
                        "val_macro_f1": 0.7,
                        "val_weighted_f1": 0.75,
                        "n_estimators": 100,
                        "analyzer": "word",
                        "max_depth": 12,
                    },
                    {
                        "preset": "balanced",
                        "val_accuracy": 0.9,
                        "val_macro_f1": 0.85,
                        "val_weighted_f1": 0.88,
                        "n_estimators": 300,
                        "analyzer": "word",
                        "max_depth": None,
                    },
                ],
                "tree_history": [
                    {
                        "n_estimators": i,
                        "oob_score": 0.5,
                        "val_accuracy": 0.6,
                        "val_macro_f1": 0.55,
                    }
                    for i in range(5, 105, 5)
                ],
                "best_preset": "balanced",
                "best_val_macro_f1": 0.85,
                "slice_metrics": {},
                "confidence": {
                    "mean_confidence": 0.8,
                    "mean_confidence_correct": 0.85,
                    "mean_confidence_incorrect": 0.4,
                    "ece": 0.1,
                    "bins": [
                        {
                            "bin": i,
                            "lo": i / 10,
                            "hi": (i + 1) / 10,
                            "n": 1,
                            "accuracy": 0.5,
                            "mean_confidence": 0.5,
                        }
                        for i in range(10)
                    ],
                },
                "top_features": [
                    {"feature": f"f{i}", "importance": 1.0 / (i + 1)} for i in range(25)
                ],
                "confusion_pairs": [{"true": "a", "pred": "b", "count": 2}],
            },
        )

    stepped = [s for s, _ in logs if s is not None]
    assert len(set(stepped)) >= 50
    assert any("train/oob_score" in d for _, d in logs)
    assert any("sweep/val_macro_f1" in d for _, d in logs)
    assert any("interp/feature_importance" in d for _, d in logs)
    assert fake_run.summary.get("wandb/logged_steps", 0) >= 50


def test_corpus_n_prefers_largest_file(tmp_path: Path):
    small = tmp_path / "documents_from_skeletons_n240_seed42.jsonl"
    large = tmp_path / "documents_from_skeletons_n2000_seed42.jsonl"
    small.write_text("{}\n", encoding="utf-8")
    large.write_text("{}\n", encoding="utf-8")
    assert corpus_n_from_path(small) == 240
    assert corpus_n_from_path(large) == 2000
    assert _largest_corpus_path([small, large]) == large


def test_ensure_seed_corpus_reuses_large_enough(tmp_path: Path, monkeypatch):
    docs_dir = tmp_path / "documents"
    noisy_dir = tmp_path / "noisy"
    docs_dir.mkdir()
    noisy_dir.mkdir()
    docs = docs_dir / "documents_from_skeletons_n2000_seed42.jsonl"
    noisy = noisy_dir / "noisy_from_documents_from_skeletons_n2000_seed42.jsonl"
    docs.write_text("{}\n", encoding="utf-8")
    noisy.write_text("{}\n", encoding="utf-8")

    class _Cfg:
        document_output_dir = docs_dir
        noisy_output_dir = noisy_dir

    monkeypatch.setattr(
        "src.classification.random_forest.Config.load",
        staticmethod(lambda: _Cfg()),
    )
    called: list[int] = []

    def _fake_run_seed(**kwargs):
        called.append(int(kwargs["n"]))
        return {"documents": "x", "noisy": "y"}

    monkeypatch.setattr(
        "src.generation.run_seed_pipeline.run_seed",
        _fake_run_seed,
    )
    paths = ensure_seed_corpus(n=2000, seed=42, log_wandb=False)
    assert paths["generated"] == "false"
    assert paths["n"] == "2000"
    assert Path(paths["documents"]) == docs
    assert called == []


def test_ensure_seed_corpus_regenerates_when_too_small(tmp_path: Path, monkeypatch):
    docs_dir = tmp_path / "documents"
    noisy_dir = tmp_path / "noisy"
    docs_dir.mkdir()
    noisy_dir.mkdir()
    (docs_dir / "documents_from_skeletons_n240_seed42.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    class _Cfg:
        document_output_dir = docs_dir
        noisy_output_dir = noisy_dir

    monkeypatch.setattr(
        "src.classification.random_forest.Config.load",
        staticmethod(lambda: _Cfg()),
    )

    def _fake_run_seed(**kwargs):
        assert kwargs["n"] == 2000
        return {"documents": str(docs_dir / "new.jsonl"), "noisy": str(noisy_dir / "new.jsonl")}

    monkeypatch.setattr(
        "src.generation.run_seed_pipeline.run_seed",
        _fake_run_seed,
    )
    monkeypatch.setattr(
        "src.utils.wandb_utils.load_wandb_settings",
        lambda enabled=False: object(),
    )
    paths = ensure_seed_corpus(n=2000, seed=42, log_wandb=False)
    assert paths["generated"] == "true"
    assert paths["n"] == "2000"


def test_notebook_exists():
    nb = Path("notebooks/random_forest_text_handwriting_classification.ipynb")
    assert nb.exists()
    text = nb.read_text(encoding="utf-8")
    assert "RandomForest" in text or "random_forest" in text
    assert "handwriting" in text.lower()
    assert "train_rf_multilayer" in text
    assert "rf-notebook-multilayer" in text
    assert "DEFAULT_PRESET_NAMES" in text
    assert "SEED_N = 2000" in text
    assert "seed_n=SEED_N" in text
