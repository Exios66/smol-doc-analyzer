"""Tests for SQL-corpus → train/test preparation helpers."""

from __future__ import annotations

from pathlib import Path

from src.storage.sample_generator import generate_corpus
from src.storage.store import DocumentStore
from src.storage.training import (
    fit_tfidf_random_forest,
    prepare_both_applications,
    prepare_classification_dataset,
    prepare_extraction_dataset,
)
from src.utils.io import load_jsonl, read_json


def test_prepare_classification_and_extraction(tmp_path: Path):
    store = DocumentStore(tmp_path / "docs.db")
    corpus = generate_corpus(
        seed=2,
        medical_per_type=2,
        salvage_per_type=2,
        bundles_per_app=0,
        include_canonical_fixtures=True,
    )
    store.bulk_upsert(corpus.documents, claims=corpus.claims)

    clf = prepare_classification_dataset(
        store, tmp_path / "clf", application="salvage_claims"
    )
    assert set(clf["labels"]) == {"log", "sales", "other"}
    assert sum(clf["counts"].values()) >= 3
    train = load_jsonl(tmp_path / "clf" / "train.jsonl")
    assert all("label_id" in r and "text" in r for r in train)

    ext = prepare_extraction_dataset(
        store, tmp_path / "ext", application="salvage_claims"
    )
    assert "vin" in ext["extraction_fields"]
    gold = load_jsonl(tmp_path / "ext" / "docie_eval.jsonl")
    assert gold and "ground_truth_fields" in gold[0]


def test_prepare_both_and_fit_rf(tmp_path: Path):
    store = DocumentStore(tmp_path / "docs.db")
    corpus = generate_corpus(
        seed=3,
        medical_per_type=3,
        salvage_per_type=3,
        bundles_per_app=1,
        include_canonical_fixtures=True,
    )
    store.bulk_upsert(corpus.documents, claims=corpus.claims)
    report = prepare_both_applications(store, tmp_path / "prepared")
    assert "medical_bills" in report["applications"]
    assert "salvage_claims" in report["applications"]

    metrics = fit_tfidf_random_forest(
        tmp_path / "prepared" / "medical_bills" / "classification",
        model_out=tmp_path / "models" / "med.joblib",
        smoke=True,
    )
    assert metrics["n_train"] > 0
    assert (tmp_path / "models" / "med.joblib").exists()
    saved = read_json(tmp_path / "models" / "med.metrics.json")
    assert "n_train" in saved


def test_notebooks_exist_and_mention_sql_store():
    names = [
        "sample_document_corpus_walkthrough.ipynb",
        "sample_corpus_sql_integrations.ipynb",
        "sample_corpus_train_test_pipeline.ipynb",
    ]
    for name in names:
        path = Path("notebooks") / name
        assert path.exists(), name
        text = path.read_text(encoding="utf-8")
        assert "DocumentStore" in text
        assert "src/storage" in text or "src.storage" in text
