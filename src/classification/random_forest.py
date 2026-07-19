"""TF-IDF + Random Forest document-type classification for the synthetic corpus.

Supports clean typed text and OCR / handwriting-style noisy variants produced by
``src.generation.noise_injection``. Designed for notebook and CLI reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json


SURFACE_TYPED = "typed"
SURFACE_HANDWRITING_OCR = "handwriting_ocr"


def _rows_from_docs(docs: list[dict[str, Any]], surface: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in docs:
        text = (doc.get("text") or "").strip()
        label = doc.get("document_type")
        if not text or not label:
            continue
        rows.append(
            {
                "record_id": doc.get("record_id"),
                "claim_id": doc.get("claim_id"),
                "text": text,
                "document_type": label,
                "surface": surface,
                "is_noisy": bool(doc.get("is_noisy", surface == SURFACE_HANDWRITING_OCR)),
            }
        )
    return rows


def load_text_handwriting_corpus(
    docs_path: Path | None = None,
    noisy_path: Path | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Load clean typed documents plus OCR/handwriting-noisy variants."""
    cfg = cfg or Config.load()
    if docs_path is None:
        candidates = sorted(cfg.document_output_dir.glob("documents_from_skeletons_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No document JSONL under {cfg.document_output_dir}. "
                "Run: python -m src.generation.run_seed_pipeline --n 240 --seed 42"
            )
        docs_path = candidates[-1]
    if noisy_path is None:
        stem = docs_path.stem
        noisy_path = cfg.noisy_output_dir / f"noisy_from_{stem}.jsonl"
        if not noisy_path.exists():
            noisy_candidates = sorted(cfg.noisy_output_dir.glob("noisy_from_*.jsonl"))
            if not noisy_candidates:
                raise FileNotFoundError(
                    f"No noisy JSONL under {cfg.noisy_output_dir}. "
                    "Run the seed pipeline (includes noise injection)."
                )
            noisy_path = noisy_candidates[-1]

    clean = _rows_from_docs(load_jsonl(docs_path), SURFACE_TYPED)
    noisy = _rows_from_docs(load_jsonl(noisy_path), SURFACE_HANDWRITING_OCR)
    frame = pd.DataFrame(clean + noisy)
    frame.attrs["docs_path"] = str(docs_path)
    frame.attrs["noisy_path"] = str(noisy_path)
    return frame


def assign_split_column(frame: pd.DataFrame, splits_path: Path | None = None) -> pd.DataFrame:
    """Attach train/val/test using project splits when available; else stratified holdout."""
    cfg = Config.load()
    path = splits_path or cfg.splits_path
    out = frame.copy()
    if path.exists():
        splits = read_json(path)
        id_to_split: dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            for rid in splits.get(split_name, []):
                id_to_split[rid] = split_name
        out["split"] = out["record_id"].map(id_to_split)
        missing = out["split"].isna()
        if missing.any():
            # Stable fallback for rows not listed in splits.json
            rng = np.random.default_rng(42)
            choices = rng.choice(["train", "val", "test"], size=int(missing.sum()), p=[0.7, 0.15, 0.15])
            out.loc[missing, "split"] = choices
    else:
        # Stratify when every class has enough rows; otherwise fall back to random split.
        counts = out["document_type"].value_counts()
        can_stratify = bool(len(out) >= 10 and counts.min() >= 2)
        stratify = out["document_type"] if can_stratify else None
        train_idx, temp_idx = train_test_split(
            out.index,
            test_size=0.3,
            random_state=42,
            stratify=stratify,
        )
        temp_labels = out.loc[temp_idx, "document_type"]
        can_stratify_temp = bool(len(temp_idx) >= 4 and temp_labels.value_counts().min() >= 2)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels if can_stratify_temp else None,
        )
        out["split"] = "train"
        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test"
    return out


def build_document_type_pipeline(
    n_estimators: int = 200,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    lowercase=True,
                    strip_accents="unicode",
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    max_depth=None,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )


def evaluate_classifier(
    model: Pipeline,
    texts: list[str] | pd.Series,
    y_true: list[str] | pd.Series,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    y_pred = model.predict(texts)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(texts)
    label_list = labels or sorted(set(list(y_true)) | set(list(y_pred)))
    report = classification_report(
        y_true, y_pred, labels=label_list, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
        "labels": label_list,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": list(y_pred),
    }
    if y_proba is not None:
        metrics["max_proba"] = y_proba.max(axis=1).tolist()
        metrics["classes"] = list(model.classes_)
    return metrics


def top_tfidf_feature_importances(model: Pipeline, top_k: int = 30) -> pd.DataFrame:
    """Map Random Forest importances back to TF-IDF feature names."""
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    forest: RandomForestClassifier = model.named_steps["rf"]
    names = np.asarray(vectorizer.get_feature_names_out())
    importances = forest.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    return pd.DataFrame(
        {
            "feature": names[order],
            "importance": importances[order],
        }
    )


def ensure_seed_corpus(
    n: int = 240,
    seed: int = 42,
    *,
    log_wandb: bool = False,
) -> dict[str, str]:
    """Generate the synthetic corpus if document JSONL files are missing.

    Nested generation does **not** open a WandB run by default so notebook / RF
    training runs are not mis-attributed as ``seed_pipeline`` experiments.
    Pass ``log_wandb=True`` (or call ``run_seed_pipeline`` directly) to track
    corpus generation separately.
    """
    cfg = Config.load()
    existing = sorted(cfg.document_output_dir.glob("documents_from_skeletons_*.jsonl"))
    if existing:
        noisy = sorted(cfg.noisy_output_dir.glob("noisy_from_*.jsonl"))
        return {
            "documents": str(existing[-1]),
            "noisy": str(noisy[-1]) if noisy else "",
            "generated": "false",
        }
    from src.generation.run_seed_pipeline import run_seed
    from src.utils.wandb_utils import load_wandb_settings

    settings = load_wandb_settings(enabled=log_wandb)
    paths = run_seed(n=n, seed=seed, skip_ingest=True, wandb_settings=settings)
    paths["generated"] = "true"
    return paths


def _label_ids(labels: Sequence[str], values: Sequence[str]) -> list[int] | None:
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    ids: list[int] = []
    for value in values:
        idx = label_to_id.get(value)
        if idx is None:
            return None
        ids.append(idx)
    return ids


def log_random_forest_to_wandb(
    *,
    doc_metrics: dict[str, Any],
    surface_metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    y_true: Sequence[str] | None = None,
    artifact_paths: Sequence[Path] | None = None,
    wandb_settings: Any | None = None,
    run_name: str | None = None,
) -> None:
    """Log RF document-type (+ optional surface) metrics to Weights & Biases."""
    from src.utils.wandb_utils import start_run

    labels = list(doc_metrics.get("labels") or [])
    preds = list(doc_metrics.get("predictions") or [])
    name = run_name or "rf-train-random-forest-classifier"
    run_config = {
        "task": "document_type_random_forest",
        "model": "sklearn.RandomForestClassifier",
        **(config or {}),
    }
    with start_run(
        name=name,
        job_type="train",
        config=run_config,
        tags=["classification", "random_forest"],
        settings=wandb_settings,
    ) as wb:
        summary: dict[str, Any] = {
            "test/accuracy": doc_metrics["accuracy"],
            "test/macro_f1": doc_metrics["macro_f1"],
            "test/weighted_f1": doc_metrics.get("weighted_f1"),
            "test/n": doc_metrics.get("n"),
        }
        if surface_metrics:
            summary["surface/accuracy"] = surface_metrics["accuracy"]
            summary["surface/macro_f1"] = surface_metrics["macro_f1"]
            summary["surface/n"] = surface_metrics.get("n")
        wb.summary(summary)
        wb.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})

        if y_true is not None and preds and labels:
            y_true_ids = _label_ids(labels, list(y_true))
            y_pred_ids = _label_ids(labels, preds)
            if y_true_ids is not None and y_pred_ids is not None:
                wb.log_confusion_matrix(
                    key="test/confusion_matrix",
                    y_true=y_true_ids,
                    y_pred=y_pred_ids,
                    class_names=labels,
                )

        report = doc_metrics.get("classification_report") or {}
        per_class_rows: list[list[Any]] = []
        for label in labels:
            stats = report.get(label) or {}
            if isinstance(stats, dict):
                per_class_rows.append(
                    [
                        label,
                        float(stats.get("precision", 0.0)),
                        float(stats.get("recall", 0.0)),
                        float(stats.get("f1-score", 0.0)),
                        int(stats.get("support", 0)),
                    ]
                )
        if per_class_rows:
            wb.log_table(
                "test/per_class",
                ["label", "precision", "recall", "f1", "support"],
                per_class_rows,
            )

        if surface_metrics:
            surface_labels = list(surface_metrics.get("labels") or [])
            surface_report = surface_metrics.get("classification_report") or {}
            surface_rows: list[list[Any]] = []
            for label in surface_labels:
                stats = surface_report.get(label) or {}
                if isinstance(stats, dict):
                    surface_rows.append(
                        [
                            label,
                            float(stats.get("precision", 0.0)),
                            float(stats.get("recall", 0.0)),
                            float(stats.get("f1-score", 0.0)),
                            int(stats.get("support", 0)),
                        ]
                    )
            if surface_rows:
                wb.log_table(
                    "surface/per_class",
                    ["label", "precision", "recall", "f1", "support"],
                    surface_rows,
                )

        if artifact_paths:
            wb.log_artifact_files(
                name=f"random-forest-eval-{name}",
                paths=artifact_paths,
                artifact_type="evaluation",
                metadata={
                    "accuracy": doc_metrics["accuracy"],
                    "macro_f1": doc_metrics["macro_f1"],
                },
            )


def save_random_forest_bundle(
    model: Pipeline,
    out_dir: Path,
    metrics: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "random_forest_pipeline.joblib"
    joblib.dump(model, model_path)
    # JSON-friendly metrics (drop raw prediction lists if huge — keep summary)
    summary = {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "n": metrics.get("n"),
        "labels": metrics.get("labels"),
        "classification_report": metrics.get("classification_report"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }
    write_json(out_dir / "eval_metrics.json", summary)
    write_json(out_dir / "train_meta.json", meta or {})
    label2id = {lab: i for i, lab in enumerate(metrics.get("labels") or list(model.classes_))}
    write_json(out_dir / "label2id.json", label2id)
    return model_path


def write_predictions_jsonl(
    path: Path,
    record_ids: list[Any],
    y_true: list[Any],
    y_pred: list[Any],
    surfaces: list[Any] | None = None,
    confidences: list[float] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, rid in enumerate(record_ids):
        row = {
            "record_id": rid,
            "true_label": y_true[i],
            "predicted_label": y_pred[i],
            "correct": y_true[i] == y_pred[i],
        }
        if surfaces is not None:
            row["surface"] = surfaces[i]
        if confidences is not None:
            row["confidence"] = float(confidences[i])
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
