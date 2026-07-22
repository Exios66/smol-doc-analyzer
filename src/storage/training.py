"""Prepare train/val/test artifacts from the sample document corpus store.

Bridges `DocumentStore` exports to classification / extraction / DICIE eval
shapes so notebooks and CLIs can fine-tune on synthetic medical + salvage docs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.docie.applications import load_application
from src.storage.store import DocumentStore
from src.utils.io import write_json, write_jsonl


def _labels_for_application(application: str) -> list[str]:
    profile = load_application(application)
    return list(profile.labels)


def prepare_classification_dataset(
    store: DocumentStore,
    out_dir: Path,
    *,
    application: str,
    include_splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    """Write train/val/test JSONL + label2id for text classification.

    Label space comes from the application taxonomy (medical_bills or
    salvage_claims), not the ACORD form categories used by Stage A.
    """
    labels = _labels_for_application(application)
    label2id = {label: i for i, label in enumerate(labels)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    label_counts: Counter[str] = Counter()
    for split in include_splits:
        docs = store.list_documents(application=application, split=split)
        rows = []
        for doc in docs:
            if doc.document_type not in label2id:
                continue
            rows.append(
                {
                    "record_id": doc.document_id,
                    "text": doc.text,
                    "label": doc.document_type,
                    "label_id": label2id[doc.document_type],
                    "application": application,
                    "claim_id": doc.claim_id,
                }
            )
            label_counts[doc.document_type] += 1
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)

    # Also emit an "all" export for tools that hash-split themselves.
    all_docs = store.list_documents(application=application)
    write_jsonl(
        out_dir / "all.jsonl",
        [
            {
                "record_id": d.document_id,
                "text": d.text,
                "label": d.document_type,
                "label_id": label2id.get(d.document_type, -1),
                "application": application,
                "claim_id": d.claim_id,
                "split": d.split,
            }
            for d in all_docs
            if d.document_type in label2id
        ],
    )

    summary = {
        "application": application,
        "labels": labels,
        "counts": counts,
        "label_counts": dict(label_counts),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "label2id.json", label2id)
    write_json(out_dir / "summary.json", summary)
    return summary


def prepare_extraction_dataset(
    store: DocumentStore,
    out_dir: Path,
    *,
    application: str,
    include_splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    """Write extraction gold JSONL (text + ground_truth_fields) per split."""
    profile = load_application(application)
    field_names = list(profile.extraction_fields)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split in include_splits:
        docs = store.list_documents(application=application, split=split)
        rows = []
        for doc in docs:
            gt = doc.ground_truth_fields()
            rows.append(
                {
                    "record_id": doc.document_id,
                    "application": application,
                    "document_type": doc.document_type,
                    "text": doc.text,
                    "claim_id": doc.claim_id,
                    "ground_truth_fields": {
                        k: gt.get(k) for k in field_names
                    },
                    "fields": {k: gt.get(k) for k in field_names},
                    "split": split,
                }
            )
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)

    # Combined DICIE-eval shaped file (all splits).
    all_rows = []
    for doc in store.list_documents(application=application):
        gt = doc.ground_truth_fields()
        all_rows.append(
            {
                "record_id": doc.document_id,
                "application": application,
                "document_type": doc.document_type,
                "text": doc.text,
                "claim_id": doc.claim_id,
                "ground_truth_fields": {k: gt.get(k) for k in field_names},
                "split": doc.split,
            }
        )
    write_jsonl(out_dir / "docie_eval.jsonl", all_rows)

    summary = {
        "application": application,
        "extraction_fields": field_names,
        "counts": counts,
        "n_all": len(all_rows),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def prepare_both_applications(
    store: DocumentStore,
    out_root: Path,
    *,
    applications: tuple[str, ...] = ("medical_bills", "salvage_claims"),
) -> dict[str, Any]:
    """Prepare classification + extraction artifacts for each application."""
    out_root = Path(out_root)
    report: dict[str, Any] = {"applications": {}}
    for app in applications:
        clf_dir = out_root / app / "classification"
        ext_dir = out_root / app / "extraction"
        report["applications"][app] = {
            "classification": prepare_classification_dataset(
                store, clf_dir, application=app
            ),
            "extraction": prepare_extraction_dataset(
                store, ext_dir, application=app
            ),
        }
    write_json(out_root / "prepare_report.json", report)
    return report


def fit_tfidf_random_forest(
    prepared_dir: Path,
    *,
    model_out: Path,
    random_state: int = 42,
    smoke: bool = True,
) -> dict[str, Any]:
    """Train a lightweight TF-IDF + RandomForest on a prepared classification dir."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.pipeline import Pipeline

    prepared_dir = Path(prepared_dir)
    label2id = json.loads((prepared_dir / "label2id.json").read_text(encoding="utf-8"))
    labels = sorted(label2id, key=lambda k: label2id[k])

    def _load(split: str) -> pd.DataFrame:
        path = prepared_dir / f"{split}.jsonl"
        if not path.exists():
            return pd.DataFrame(columns=["text", "label"])
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)

    train_df = _load("train")
    val_df = _load("val")
    test_df = _load("test")
    if train_df.empty:
        raise RuntimeError(f"No train rows in {prepared_dir}")

    n_estimators = 64 if smoke else 200
    max_features = 8000 if smoke else 20000
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=max_features,
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
                ),
            ),
        ]
    )
    pipe.fit(train_df["text"], train_df["label"])

    metrics: dict[str, Any] = {
        "application_labels": labels,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "smoke": smoke,
    }
    for name, df in (("val", val_df), ("test", test_df)):
        if df.empty:
            continue
        pred = pipe.predict(df["text"])
        metrics[f"{name}_accuracy"] = float(accuracy_score(df["label"], pred))
        metrics[f"{name}_macro_f1"] = float(
            f1_score(df["label"], pred, average="macro", zero_division=0)
        )
        metrics[f"{name}_report"] = classification_report(
            df["label"], pred, labels=labels, zero_division=0, output_dict=True
        )

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "labels": labels, "label2id": label2id, "metrics": metrics},
        model_out,
    )
    write_json(model_out.with_suffix(".metrics.json"), metrics)
    return metrics
