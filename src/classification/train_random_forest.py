"""CLI: train TF-IDF + Random Forest on typed + handwriting/OCR corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.classification.random_forest import (
    SURFACE_HANDWRITING_OCR,
    SURFACE_TYPED,
    assign_split_column,
    build_document_type_pipeline,
    ensure_seed_corpus,
    evaluate_classifier,
    load_text_handwriting_corpus,
    log_random_forest_to_wandb,
    save_random_forest_bundle,
    write_predictions_jsonl,
)
from src.utils.config import Config
from src.utils.provenance import ProvenanceRecord, log_provenance
from src.utils.wandb_utils import add_wandb_cli_flags, settings_from_args

logger = logging.getLogger(__name__)


def train(
    cfg: Config | None = None,
    docs_path: Path | None = None,
    noisy_path: Path | None = None,
    out_dir: Path | None = None,
    n_estimators: int = 300,
    seed_n: int = 240,
    ensure_data: bool = True,
    wandb_settings=None,
    wandb_run_name: str | None = None,
) -> Path:
    cfg = cfg or Config.load()
    if ensure_data:
        # Keep seed generation out of this experiment's WandB run.
        ensure_seed_corpus(n=seed_n, seed=42, log_wandb=False)

    frame = load_text_handwriting_corpus(docs_path=docs_path, noisy_path=noisy_path, cfg=cfg)
    frame = assign_split_column(frame)
    train_df = frame[frame["split"] == "train"]
    val_df = frame[frame["split"] == "val"]
    test_df = frame[frame["split"] == "test"]
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    model = build_document_type_pipeline(n_estimators=n_estimators, random_state=42)
    model.fit(fit_df["text"], fit_df["document_type"])
    metrics = evaluate_classifier(
        model, test_df["text"], test_df["document_type"], labels=list(model.classes_)
    )

    surface_model = build_document_type_pipeline(
        n_estimators=max(100, n_estimators // 2), random_state=42
    )
    surface_model.fit(fit_df["text"], fit_df["surface"])
    surface_metrics = evaluate_classifier(
        surface_model,
        test_df["text"],
        test_df["surface"],
        labels=[SURFACE_TYPED, SURFACE_HANDWRITING_OCR],
    )

    out = out_dir or (cfg.models_dir / "random_forest_classifier")
    meta = {
        "task": "document_type_random_forest",
        "docs_path": frame.attrs.get("docs_path"),
        "noisy_path": frame.attrs.get("noisy_path"),
        "n_fit": int(len(fit_df)),
        "n_test": int(len(test_df)),
        "n_estimators": n_estimators,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "surface_test_accuracy": surface_metrics["accuracy"],
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

    report = cfg.evaluation_reports_dir / "random_forest_classification_report.md"
    cfg.evaluation_reports_dir.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Random Forest classification report",
                "",
                f"- Fit rows: {len(fit_df)}",
                f"- Test rows: {len(test_df)}",
                f"- Document-type accuracy: **{metrics['accuracy']:.4f}**",
                f"- Document-type macro F1: **{metrics['macro_f1']:.4f}**",
                f"- Surface accuracy: **{surface_metrics['accuracy']:.4f}**",
                f"- Model: `models/random_forest_classifier/{model_path.name}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    log_random_forest_to_wandb(
        doc_metrics=metrics,
        surface_metrics=surface_metrics,
        config={
            "n_estimators": n_estimators,
            "n_fit": int(len(fit_df)),
            "n_test": int(len(test_df)),
            "docs_path": frame.attrs.get("docs_path"),
            "noisy_path": frame.attrs.get("noisy_path"),
            "out_dir": str(out),
        },
        y_true=test_df["document_type"].tolist(),
        artifact_paths=[
            out / "eval_metrics.json",
            out / "train_meta.json",
            report,
        ],
        wandb_settings=wandb_settings,
        run_name=wandb_run_name or f"rf-train-{out.name}",
    )

    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"rf-train-{out.name}",
            stage="classification_train_random_forest",
            source=str(frame.attrs.get("docs_path")),
            prompt_version="random_forest_v1",
            model="sklearn.RandomForestClassifier",
            extra=meta,
        ),
    )
    logger.info(
        "RF accuracy=%.4f macro_f1=%.4f surface_acc=%.4f -> %s",
        metrics["accuracy"],
        metrics["macro_f1"],
        surface_metrics["accuracy"],
        out,
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=None)
    parser.add_argument("--noisy", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed-n", type=int, default=240)
    parser.add_argument("--no-ensure-data", action="store_true")
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
        )
    )


if __name__ == "__main__":
    main()
