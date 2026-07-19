"""CLI: multilayer TF-IDF + Random Forest train/eval with WandB logging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd

from src.classification.random_forest import (
    DEFAULT_PRESET_NAMES,
    SURFACE_HANDWRITING_OCR,
    SURFACE_TYPED,
    assign_split_column,
    confidence_diagnostics,
    ensure_seed_corpus,
    evaluate_by_surface,
    evaluate_classifier,
    fit_pipeline_with_tree_curve,
    load_text_handwriting_corpus,
    log_random_forest_to_wandb,
    pipeline_from_preset,
    profile_corpus,
    save_random_forest_bundle,
    set_vectorizer_min_df,
    top_confusion_pairs,
    top_tfidf_feature_importances,
    write_predictions_jsonl,
)
from src.utils.config import Config
from src.utils.io import write_json
from src.utils.provenance import ProvenanceRecord, log_provenance
from src.utils.wandb_utils import add_wandb_cli_flags, settings_from_args

logger = logging.getLogger(__name__)


def _parse_presets(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return list(DEFAULT_PRESET_NAMES)
    names = [p.strip() for p in raw.split(",") if p.strip()]
    return names or list(DEFAULT_PRESET_NAMES)


def run_capacity_sweep(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preset_names: Sequence[str],
    *,
    smoke: bool = False,
    random_state: int = 42,
) -> tuple[str, list[dict[str, Any]]]:
    """Layer 1: fit each preset on train, score on val; return best preset + rows."""
    if val_df.empty:
        # Degenerate split: score on train and still pick a preset.
        val_df = train_df
    rows: list[dict[str, Any]] = []
    for name in preset_names:
        model, cfg = pipeline_from_preset(name, smoke=smoke, random_state=random_state)
        if smoke:
            set_vectorizer_min_df(model, 1)
        model.fit(train_df["text"], train_df["document_type"])
        metrics = evaluate_classifier(
            model,
            val_df["text"],
            val_df["document_type"],
            labels=list(model.classes_),
        )
        rows.append(
            {
                "preset": name,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_weighted_f1": metrics["weighted_f1"],
                "n_estimators": cfg.get("n_estimators"),
                "analyzer": cfg.get("analyzer"),
                "max_depth": cfg.get("max_depth"),
                "max_features": cfg.get("max_features"),
                "description": cfg.get("description"),
            }
        )
        logger.info(
            "sweep preset=%s val_acc=%.4f val_macro_f1=%.4f",
            name,
            metrics["accuracy"],
            metrics["macro_f1"],
        )
    best = max(rows, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]))
    return str(best["preset"]), rows


def train(
    cfg: Config | None = None,
    docs_path: Path | None = None,
    noisy_path: Path | None = None,
    out_dir: Path | None = None,
    n_estimators: int = 300,
    seed_n: int = 400,
    ensure_data: bool = True,
    wandb_settings=None,
    wandb_run_name: str | None = None,
    presets: Sequence[str] | None = None,
    smoke: bool = False,
) -> Path:
    """Multilayer RF train: corpus → sweep → best → dual → slices → WandB."""
    cfg = cfg or Config.load()
    if ensure_data:
        ensure_seed_corpus(n=seed_n, seed=42, log_wandb=False)

    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path, cfg=cfg)
    frame = assign_split_column(frame)
    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    val_df = frame[frame["split"] == "val"].reset_index(drop=True)
    test_df = frame[frame["split"] == "test"].reset_index(drop=True)
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    # Layer 0 — corpus profile
    data_profile = profile_corpus(frame)

    preset_names = list(presets) if presets is not None else list(DEFAULT_PRESET_NAMES)
    # Legacy --n-estimators only affects non-smoke balanced when it is the sole preset;
    # multilayer sweep uses named presets. Keep param for CLI compatibility.
    _ = n_estimators

    # Layer 1 — capacity sweep on val
    best_preset, sweep_rows = run_capacity_sweep(
        train_df if not train_df.empty else fit_df,
        val_df if not val_df.empty else test_df,
        preset_names,
        smoke=smoke,
    )
    best_val_macro = next(
        (r["val_macro_f1"] for r in sweep_rows if r["preset"] == best_preset),
        None,
    )

    # Layer 2 — refit winner on train+val with tree-growth curve; dual surface head
    model, best_cfg = pipeline_from_preset(best_preset, smoke=smoke)
    if smoke:
        set_vectorizer_min_df(model, 1)
    # ~20 curve points for full runs (so WandB charts aren't a single dot).
    target_trees = int(best_cfg.get("n_estimators") or 100)
    tree_chunk = 5 if smoke else max(5, target_trees // 20)
    holdout_for_curve = val_df if not val_df.empty else test_df
    tree_history = fit_pipeline_with_tree_curve(
        model,
        fit_df["text"],
        fit_df["document_type"],
        val_texts=holdout_for_curve["text"],
        val_y=holdout_for_curve["document_type"],
        tree_chunk=tree_chunk,
    )
    metrics = evaluate_classifier(
        model, test_df["text"], test_df["document_type"], labels=list(model.classes_)
    )

    surface_model, _ = pipeline_from_preset(
        "shallow" if "shallow" in preset_names else best_preset,
        smoke=smoke,
    )
    if smoke:
        set_vectorizer_min_df(surface_model, 1)
    surface_model.fit(fit_df["text"], fit_df["surface"])
    surface_metrics = evaluate_classifier(
        surface_model,
        test_df["text"],
        test_df["surface"],
        labels=[SURFACE_TYPED, SURFACE_HANDWRITING_OCR],
    )

    # Layer 3 — slice by surface
    slice_metrics = evaluate_by_surface(model, test_df)

    # Layer 4 — confidence diagnostics
    conf = confidence_diagnostics(
        test_df["document_type"].tolist(),
        metrics["predictions"],
        metrics.get("max_proba") or [],
    )

    # Layer 5 — interpretability
    fi = top_tfidf_feature_importances(model, top_k=25)
    top_features = fi.to_dict(orient="records")
    confusion_pairs = top_confusion_pairs(metrics, top_k=15)

    out = out_dir or (cfg.models_dir / "random_forest_classifier")
    meta = {
        "task": "document_type_random_forest_multilayer",
        "docs_path": frame.attrs.get("docs_path"),
        "noisy_path": frame.attrs.get("noisy_path"),
        "n_fit": int(len(fit_df)),
        "n_test": int(len(test_df)),
        "best_preset": best_preset,
        "best_preset_config": best_cfg,
        "best_val_macro_f1": best_val_macro,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "surface_test_accuracy": surface_metrics["accuracy"],
        "slice_typed_accuracy": (slice_metrics.get(SURFACE_TYPED) or {}).get("accuracy"),
        "slice_ocr_accuracy": (slice_metrics.get(SURFACE_HANDWRITING_OCR) or {}).get(
            "accuracy"
        ),
        "confidence_ece": conf.get("ece"),
        "smoke": smoke,
        "presets": list(preset_names),
    }
    model_path = save_random_forest_bundle(model, out, metrics, meta=meta)
    write_predictions_jsonl(
        out / "test_predictions.jsonl",
        record_ids=test_df["record_id"].tolist(),
        y_true=test_df["document_type"].tolist(),
        y_pred=metrics["predictions"],
        surfaces=test_df["surface"].tolist(),
        confidences=metrics.get("max_proba"),
    )
    joblib.dump(surface_model, out / "surface_random_forest_pipeline.joblib")
    write_json(out / "sweep_results.json", {"best_preset": best_preset, "rows": sweep_rows})
    write_json(
        out / "layer_diagnostics.json",
        {
            "data_profile": data_profile,
            "tree_history": tree_history,
            "slice_metrics": {
                k: {
                    "accuracy": v.get("accuracy"),
                    "macro_f1": v.get("macro_f1"),
                    "weighted_f1": v.get("weighted_f1"),
                    "n": v.get("n"),
                }
                for k, v in slice_metrics.items()
            },
            "confidence": conf,
            "top_features": top_features,
            "confusion_pairs": confusion_pairs,
        },
    )

    typed_acc = (slice_metrics.get(SURFACE_TYPED) or {}).get("accuracy")
    ocr_acc = (slice_metrics.get(SURFACE_HANDWRITING_OCR) or {}).get("accuracy")
    report = cfg.evaluation_reports_dir / "random_forest_classification_report.md"
    cfg.evaluation_reports_dir.mkdir(parents=True, exist_ok=True)
    sweep_lines = [
        "| preset | val_accuracy | val_macro_f1 | n_estimators | analyzer |",
        "|---|---:|---:|---:|---|",
    ]
    for r in sweep_rows:
        sweep_lines.append(
            f"| {r['preset']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | "
            f"{r['n_estimators']} | {r['analyzer']} |"
        )
    report.write_text(
        "\n".join(
            [
                "# Random Forest multilayer classification report",
                "",
                f"- Fit rows: {len(fit_df)}",
                f"- Test rows: {len(test_df)}",
                f"- Best preset (by val macro F1): **{best_preset}**",
                f"- Document-type test accuracy: **{metrics['accuracy']:.4f}**",
                f"- Document-type test macro F1: **{metrics['macro_f1']:.4f}**",
                f"- Surface accuracy: **{surface_metrics['accuracy']:.4f}**",
                f"- Slice typed accuracy: **{(typed_acc if typed_acc is not None else float('nan')):.4f}**",
                f"- Slice OCR accuracy: **{(ocr_acc if ocr_acc is not None else float('nan')):.4f}**",
                f"- Confidence ECE: **{conf.get('ece', 0.0):.4f}**",
                f"- Model: `models/random_forest_classifier/{model_path.name}`",
                "",
                "## Capacity sweep (Layer 1)",
                "",
                *sweep_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )

    layer_payload = {
        "data_profile": data_profile,
        "sweep_rows": sweep_rows,
        "tree_history": tree_history,
        "best_preset": best_preset,
        "best_val_macro_f1": best_val_macro,
        "slice_metrics": slice_metrics,
        "confidence": conf,
        "top_features": top_features,
        "confusion_pairs": confusion_pairs,
    }
    log_random_forest_to_wandb(
        doc_metrics=metrics,
        surface_metrics=surface_metrics,
        config={
            "best_preset": best_preset,
            "presets": list(preset_names),
            "smoke": smoke,
            "n_fit": int(len(fit_df)),
            "n_test": int(len(test_df)),
            "docs_path": frame.attrs.get("docs_path"),
            "noisy_path": frame.attrs.get("noisy_path"),
            "out_dir": str(out),
            **{f"preset/{k}": v for k, v in best_cfg.items() if isinstance(v, (int, float, str, bool)) or v is None},
        },
        y_true=test_df["document_type"].tolist(),
        artifact_paths=[
            out / "eval_metrics.json",
            out / "train_meta.json",
            out / "sweep_results.json",
            out / "layer_diagnostics.json",
            report,
        ],
        wandb_settings=wandb_settings,
        run_name=wandb_run_name or f"rf-multilayer-{out.name}",
        layer_payload=layer_payload,
    )

    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"rf-train-{out.name}",
            stage="classification_train_random_forest",
            source=str(frame.attrs.get("docs_path")),
            prompt_version="random_forest_multilayer_v1",
            model="sklearn.RandomForestClassifier",
            extra=meta,
        ),
    )
    logger.info(
        "RF multilayer best=%s acc=%.4f macro_f1=%.4f typed=%.4f ocr=%.4f ece=%.4f -> %s",
        best_preset,
        metrics["accuracy"],
        metrics["macro_f1"],
        typed_acc or 0.0,
        ocr_acc or 0.0,
        conf.get("ece", 0.0),
        out,
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=None)
    parser.add_argument("--noisy", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Legacy; multilayer sweep uses --presets instead",
    )
    parser.add_argument("--seed-n", type=int, default=240)
    parser.add_argument("--no-ensure-data", action="store_true")
    parser.add_argument(
        "--presets",
        type=str,
        default=",".join(DEFAULT_PRESET_NAMES),
        help="Comma-separated capacity presets (default: all four)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny estimators / vocab for CI smoke tests",
    )
    add_wandb_cli_flags(parser)
    args = parser.parse_args()
    print(
        train(
            docs_path=args.docs,
            noisy_path=args.noisy,
            out_dir=args.out,
            n_estimators=args.n_estimators,
            seed_n=args.seed_n,
            ensure_data=not args.no_ensure_data,
            wandb_settings=settings_from_args(args),
            wandb_run_name=args.wandb_run_name,
            presets=_parse_presets(args.presets),
            smoke=args.smoke,
        )
    )


if __name__ == "__main__":
    main()
