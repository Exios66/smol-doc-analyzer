"""Tests for the sample medical + salvage document corpus store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.storage.sample_generator import (
    generate_corpus,
    generate_medical_document,
    generate_salvage_document,
)
from src.storage.store import DocumentStore
from src.utils.io import load_jsonl, write_jsonl

FIXTURES = Path(__file__).parent / "fixtures" / "sample_docie_documents.jsonl"


@pytest.fixture()
def store(tmp_path: Path) -> DocumentStore:
    return DocumentStore(tmp_path / "documents.db")


def test_schema_creates_and_summarizes_empty(store: DocumentStore):
    summary = store.summary()
    assert summary["documents"] == 0
    assert summary["claims"] == 0
    assert summary["schema_version"] == 1


def test_seed_corpus_contains_medical_and_salvage(store: DocumentStore):
    corpus = generate_corpus(
        seed=7,
        medical_per_type=2,
        salvage_per_type=2,
        bundles_per_app=1,
        include_canonical_fixtures=True,
    )
    n = store.bulk_upsert(corpus.documents, claims=corpus.claims)
    assert n == len(corpus.documents)
    summary = store.summary()
    assert summary["documents"] == n
    assert summary["claims"] >= 1

    apps = {row["application"] for row in summary["by_application_type"]}
    assert "medical_bills" in apps
    assert "salvage_claims" in apps

    types = {
        (row["application"], row["document_type"])
        for row in summary["by_application_type"]
    }
    assert ("medical_bills", "hcfa") in types
    assert ("medical_bills", "ub04") in types
    assert ("salvage_claims", "log") in types
    assert ("salvage_claims", "sales") in types


def test_canonical_fixtures_roundtrip(store: DocumentStore):
    corpus = generate_corpus(
        seed=1,
        medical_per_type=0,
        salvage_per_type=0,
        bundles_per_app=0,
        include_canonical_fixtures=True,
    )
    store.bulk_upsert(corpus.documents, claims=corpus.claims)

    log = store.get_document("sal-log-001")
    assert log is not None
    assert log.document_type == "log"
    assert log.ground_truth_fields()["vin"] == "1HGCM82633A004352"
    assert "LETTER OF GUARANTEE" in log.text
    assert "American Family" in (
        store.get_claim(log.claim_id).carrier_name if log.claim_id else ""
    )

    hcfa = store.get_document("med-hcfa-001")
    assert hcfa is not None
    assert hcfa.ground_truth_fields()["claim_id"] == "CLM-2024-551122"
    assert "American Family" in hcfa.text


def test_export_docie_shape(store: DocumentStore, tmp_path: Path):
    corpus = generate_corpus(
        seed=3,
        medical_per_type=1,
        salvage_per_type=1,
        bundles_per_app=0,
        include_canonical_fixtures=True,
    )
    store.bulk_upsert(corpus.documents, claims=corpus.claims)
    out = tmp_path / "docie.jsonl"
    n = store.export_jsonl(out, format="docie", application="salvage_claims")
    assert n >= 3
    rows = load_jsonl(out)
    assert all("record_id" in r and "text" in r for r in rows)
    assert all("ground_truth_fields" in r for r in rows)
    assert all(r["application"] == "salvage_claims" for r in rows)


def test_export_classification_and_extraction(store: DocumentStore, tmp_path: Path):
    corpus = generate_corpus(
        seed=4,
        medical_per_type=1,
        salvage_per_type=0,
        bundles_per_app=0,
        include_canonical_fixtures=False,
    )
    store.bulk_upsert(corpus.documents, claims=corpus.claims)

    clf = tmp_path / "clf.jsonl"
    ext = tmp_path / "ext.jsonl"
    assert store.export_jsonl(clf, format="classification") >= 1
    assert store.export_jsonl(ext, format="extraction") >= 1
    clf_rows = load_jsonl(clf)
    ext_rows = load_jsonl(ext)
    assert "label" in clf_rows[0]
    assert "fields" in ext_rows[0]


def test_import_existing_fixtures(store: DocumentStore):
    n = store.import_docie_jsonl(FIXTURES, source_kind="test_fixture")
    assert n == 6
    assert store.count_documents(application="medical_bills") == 3
    assert store.count_documents(application="salvage_claims") == 3
    sales = store.get_document("sal-sales-002")
    assert sales is not None
    assert sales.document_type == "sales"


def test_generated_log_contains_granular_fields():
    import random

    rng = random.Random(99)
    claim, doc = generate_salvage_document(rng, document_type="log", index=1)
    assert claim.application == "salvage_claims"
    assert "LETTER OF GUARANTEE" in doc.text
    assert "Payoff Amount:" in doc.text
    assert doc.skeleton["vehicle"]["vin"]
    assert doc.skeleton["financials"]["payoff_amount"] is not None
    assert doc.ground_truth_fields()["make"]


def test_generated_hcfa_contains_carrier_and_patient():
    import random

    rng = random.Random(101)
    claim, doc = generate_medical_document(rng, document_type="hcfa", index=1)
    assert claim.application == "medical_bills"
    assert "CMS-1500" in doc.text or "HCFA" in doc.text
    assert "Carrier Name:" in doc.text
    assert doc.ground_truth_fields()["patient_id"]
    assert doc.skeleton["provider"]["npi"]


def test_claim_bundle_shares_claim_id(store: DocumentStore):
    from src.storage.sample_generator import generate_claim_bundle
    import random

    rng = random.Random(5)
    claim, docs = generate_claim_bundle(
        rng, application="salvage_claims", bundle_index=0
    )
    store.upsert_claim(claim)
    for d in docs:
        store.upsert_document(d)
    assert len(docs) == 3
    assert {d.claim_id for d in docs} == {claim.claim_id}
    listed = store.list_documents(claim_id=claim.claim_id)
    assert len(listed) == 3


def test_cli_seed_and_export(tmp_path: Path):
    from src.storage.__main__ import main

    db = tmp_path / "cli.db"
    export = tmp_path / "out.jsonl"
    rc = main(
        [
            "--db",
            str(db),
            "seed",
            "--seed",
            "11",
            "--medical-per-type",
            "1",
            "--salvage-per-type",
            "1",
            "--bundles-per-app",
            "0",
        ]
    )
    assert rc == 0
    assert db.exists()

    rc = main(["--db", str(db), "export", "--format", "docie", "--out", str(export)])
    assert rc == 0
    rows = load_jsonl(export)
    assert len(rows) >= 6

    rc = main(["--db", str(db), "summary"])
    assert rc == 0


def test_json_schemas_exist_and_parse():
    med = Path("data/schemas/medical_bill_skeleton.schema.json")
    sal = Path("data/schemas/salvage_document_skeleton.schema.json")
    assert med.exists()
    assert sal.exists()
    assert json.loads(med.read_text())["title"] == "MedicalBillSkeleton"
    assert json.loads(sal.read_text())["title"] == "SalvageDocumentSkeleton"


def test_import_export_roundtrip_preserves_fields(store: DocumentStore, tmp_path: Path):
    src = tmp_path / "in.jsonl"
    write_jsonl(
        src,
        [
            {
                "record_id": "roundtrip-1",
                "application": "salvage_claims",
                "document_type": "log",
                "text": "LETTER OF GUARANTEE\nClaim Number: CLM-9\nVIN: 1HGCM82633A004352\nYear: 2018\nMake: Honda\nModel: Accord",
                "ground_truth_fields": {
                    "claim_id": "CLM-9",
                    "vin": "1HGCM82633A004352",
                    "year": "2018",
                    "make": "Honda",
                    "model": "Accord",
                },
            }
        ],
    )
    store.import_docie_jsonl(src)
    out = tmp_path / "out.jsonl"
    store.export_jsonl(out, format="docie")
    rows = load_jsonl(out)
    assert rows[0]["ground_truth_fields"]["vin"] == "1HGCM82633A004352"


def test_upsert_document_replaces_stale_ground_truth_fields(store: DocumentStore):
    from src.storage.types import DocumentRecord, FieldRecord

    doc = DocumentRecord(
        document_id="sal-upsert-001",
        application="salvage_claims",
        document_type="log",
        text="LETTER OF GUARANTEE\nClaim Number: CLM-1\nVIN: 1HGCM82633A004352\n",
        fields=[
            FieldRecord("claim_id", "CLM-1", "ground_truth"),
            FieldRecord("vin", "1HGCM82633A004352", "ground_truth"),
            FieldRecord("year", "2018", "ground_truth"),
            FieldRecord("vin", "EXTRACTED-VIN", "extracted"),
        ],
    )
    store.upsert_document(doc)
    got = store.get_document("sal-upsert-001")
    assert got is not None
    assert got.ground_truth_fields()["year"] == "2018"
    assert any(f.field_role == "extracted" and f.field_name == "vin" for f in got.fields)

    # Re-import corrects GT and omits year — year must not linger; extracted stays.
    corrected = DocumentRecord(
        document_id="sal-upsert-001",
        application="salvage_claims",
        document_type="log",
        text=doc.text,
        fields=[
            FieldRecord("claim_id", "CLM-1", "ground_truth"),
            FieldRecord("vin", "1HGCM82633A004352", "ground_truth"),
        ],
    )
    store.upsert_document(corrected)
    got2 = store.get_document("sal-upsert-001")
    assert got2 is not None
    gt = got2.ground_truth_fields()
    assert set(gt) == {"claim_id", "vin"}
    assert "year" not in gt
    assert any(f.field_role == "extracted" and f.field_value == "EXTRACTED-VIN" for f in got2.fields)


def test_set_fields_replace_clears_omitted_keys(store: DocumentStore):
    from src.storage.types import DocumentRecord, FieldRecord

    store.upsert_document(
        DocumentRecord(
            document_id="med-set-001",
            application="medical_bills",
            document_type="hcfa",
            text="HCFA\nPatient Name: A\n",
            fields=[
                FieldRecord("claim_id", "CLM-A", "ground_truth"),
                FieldRecord("name", "A", "ground_truth"),
            ],
        )
    )
    store.set_fields("med-set-001", {"claim_id": "CLM-B"}, role="ground_truth", replace=True)
    gt = store.get_document("med-set-001").ground_truth_fields()
    assert gt == {"claim_id": "CLM-B"}
