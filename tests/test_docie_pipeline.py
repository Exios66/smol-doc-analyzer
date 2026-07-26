"""Tests for the paper Fig. 1 DICIE pipeline."""

from __future__ import annotations

from pathlib import Path

from src.docie.aggregate import aggregate_prediction, push_downstream
from src.docie.applications import list_applications, load_application
from src.docie.classify import aggregate_page_predictions, classify_document, classify_page_text
from src.docie.extract import extract_information, heuristic_extract
from src.docie.pipeline import DociePipeline, process_document, run_file
from src.docie.processing import process_document_input
from src.docie.types import PageClassification
from src.utils.config import Config
from src.utils.io import load_jsonl

FIXTURES = Path(__file__).parent / "fixtures" / "sample_docie_documents.jsonl"


def test_application_profiles_load():
    apps = list_applications()
    assert "medical_bills" in apps
    assert "salvage_claims" in apps
    medical = load_application("medical_bills")
    assert set(medical.labels) == {"hcfa", "ub04", "other"}
    assert "claim_id" in medical.extraction_fields
    assert "patient_id" in medical.extraction_fields
    salvage = load_application("salvage_claims")
    assert set(salvage.labels) == {"log", "sales", "other"}
    assert "vin" in salvage.extraction_fields


def test_heuristic_classify_medical_and_salvage():
    medical = load_application("medical_bills")
    salvage = load_application("salvage_claims")
    rows = load_jsonl(FIXTURES)

    hcfa = next(r for r in rows if r["record_id"] == "med-hcfa-001")
    pred = classify_page_text(hcfa["text"], medical)
    assert pred.label == "hcfa"
    assert pred.confidence > 0.5

    ub = next(r for r in rows if r["record_id"] == "med-ub04-002")
    pred = classify_page_text(ub["text"], medical)
    assert pred.label == "ub04"

    log = next(r for r in rows if r["record_id"] == "sal-log-001")
    pred = classify_page_text(log["text"], salvage)
    assert pred.label == "log"

    sales = next(r for r in rows if r["record_id"] == "sal-sales-002")
    pred = classify_page_text(sales["text"], salvage)
    assert pred.label == "sales"


def test_aggregate_prefers_non_other_on_tie():
    profile = load_application("salvage_claims")
    pages = [
        PageClassification(0, "other", 0.5, "heuristic_text"),
        PageClassification(1, "log", 0.5, "heuristic_text"),
    ]
    result = aggregate_page_predictions(pages, profile)
    assert result.label == "log"


def test_heuristic_extract_salvage_fields():
    text = (
        "LETTER OF GUARANTEE\n"
        "Claim Number: CLM-2024-100200\n"
        "VIN: 1HGCM82633A004352\n"
        "Year: 2018\n"
        "Make: Honda\n"
        "Model: Accord\n"
    )
    fields = heuristic_extract(
        text, ["claim_id", "vin", "year", "make", "model"]
    )
    assert fields["claim_id"][0] == "CLM-2024-100200"
    assert fields["vin"][0] == "1HGCM82633A004352"
    assert fields["year"][0] == "2018"
    assert fields["make"][0].lower().startswith("honda")
    assert "accord" in fields["model"][0].lower()


def test_heuristic_extract_medical_fields():
    text = (
        "HCFA CMS-1500\n"
        "Patient Name: Jane Q Public\n"
        "Date of Birth: 03/14/1988\n"
        "Patient ID: PID-778812\n"
        "Claim Number: CLM-2024-551122\n"
        "Address: 100 Oak Avenue, Madison WI 53703\n"
    )
    fields = heuristic_extract(
        text, ["claim_id", "name", "dob", "patient_id", "address"]
    )
    assert fields["claim_id"][0] == "CLM-2024-551122"
    assert "Jane" in fields["name"][0]
    assert fields["dob"][0] == "03/14/1988"
    assert fields["patient_id"][0] == "PID-778812"


def test_stage1_text_renders_page_image(tmp_path: Path):
    processed = process_document_input(
        record_id="stage1-text",
        text="LETTER OF GUARANTEE\nClaim Number: CLM-1\nVIN: 1HGCM82633A004352\n",
        cache_dir=tmp_path / "pages",
        application="salvage_claims",
        run_ocr=False,
    )
    assert processed.source_kind == "text"
    assert len(processed.pages) == 1
    assert processed.pages[0].image_path.exists()
    assert processed.pages[0].grayscale is True
    assert processed.pages[0].dpi == 300
    assert "CLM-1" in processed.full_text


def test_stage1_pdf_to_page_images(tmp_path: Path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "log.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(72, 720, "LETTER OF GUARANTEE")
    c.drawString(72, 700, "Claim Number: CLM-PDF-9")
    c.drawString(72, 680, "VIN: 1HGCM82633A004352")
    c.drawString(72, 660, "Year: 2019")
    c.drawString(72, 640, "Make: Toyota")
    c.drawString(72, 620, "Model: Camry")
    c.save()

    processed = process_document_input(
        record_id="stage1-pdf",
        pdf_path=pdf_path,
        cache_dir=tmp_path / "pages",
        application="salvage_claims",
        run_ocr=False,
        dpi=150,
    )
    assert processed.source_kind == "pdf"
    assert len(processed.pages) == 1
    assert processed.pages[0].image_path.exists()
    assert "CLM-PDF-9" in processed.full_text or "LETTER" in processed.full_text.upper()


def test_end_to_end_salvage_log_chain():
    cfg = Config.load()
    text = next(r for r in load_jsonl(FIXTURES) if r["record_id"] == "sal-log-001")["text"]
    result = process_document(
        application="salvage_claims",
        record_id="e2e-log",
        text=text,
        cfg=cfg,
        run_ocr=False,
    )
    assert result["application"] == "salvage_claims"
    assert result["classification"]["label"] == "log"
    assert result["document_type"] == "log"
    assert result["extraction"]["fields_flat"]["claim_id"] == "CLM-2024-100200"
    assert result["extraction"]["fields_flat"]["vin"] == "1HGCM82633A004352"
    assert result["processing"]["n_pages"] == 1
    assert "document_processing" in result["stage_timings_ms"]
    assert "document_classification" in result["stage_timings_ms"]
    assert "information_extraction" in result["stage_timings_ms"]


def test_end_to_end_medical_hcfa_chain():
    cfg = Config.load()
    text = next(r for r in load_jsonl(FIXTURES) if r["record_id"] == "med-hcfa-001")["text"]
    pipe = DociePipeline(application="medical_bills", cfg=cfg, run_ocr=False)
    prediction = pipe.process(record_id="e2e-hcfa", text=text)
    assert prediction.classification.label == "hcfa"
    assert prediction.extraction.fields_flat["claim_id"] == "CLM-2024-551122"
    assert prediction.extraction.fields_flat["name"]
    payload = prediction.response_payload()
    assert payload["document_type"] == "hcfa"
    assert "fields" in payload


def test_run_file_writes_outputs(tmp_path: Path):
    cfg = Config.load()
    out = tmp_path / "docie_out.jsonl"
    path = run_file(
        FIXTURES,
        out,
        application="salvage_claims",
        cfg=cfg,
        limit=3,
        run_ocr=False,
    )
    assert path.exists()
    rows = load_jsonl(path)
    assert len(rows) == 3
    summary = path.with_suffix(".summary.json")
    assert summary.exists()
    review = path.with_name(path.stem + ".human_review.jsonl")
    assert review.exists()


def test_full_chain_on_processed_document(tmp_path: Path):
    profile = load_application("salvage_claims")
    processed = process_document_input(
        record_id="chain",
        text=load_jsonl(FIXTURES)[3]["text"],  # sal-log-001
        cache_dir=tmp_path / "pages",
        application="salvage_claims",
        run_ocr=False,
    )
    classification = classify_document(processed, profile)
    extraction = extract_information(
        processed, profile, document_type=classification.label
    )
    prediction = aggregate_prediction(
        processed=processed,
        classification=classification,
        extraction=extraction,
        profile=profile,
    )
    assert prediction.classification.label == "log"
    assert not prediction.needs_human_review or "vin" in prediction.extraction.fields_flat

    sunk: list[dict] = []
    payload = push_downstream(prediction, sink=sunk.append)
    assert sunk and sunk[0]["document_type"] == "log"
    assert payload["fields"]["vin"]


def test_pipeline_stage_order_contract():
    """Fig. 1 stage order is fixed: process → classify → extract → aggregate."""
    cfg = Config.load()
    pipe = DociePipeline(application="salvage_claims", cfg=cfg, run_ocr=False)
    prediction = pipe.process(
        record_id="order",
        text="LETTER OF GUARANTEE\nClaim Number: CLM-9\nVIN: 1HGCM82633A004352\nYear: 2017\nMake: Ford\nModel: Escape\n",
    )
    # Timings keys document the chronological stage order
    assert list(prediction.stage_timings_ms) == [
        "document_processing",
        "document_classification",
        "information_extraction",
        "output_aggregation",
    ]


def test_heuristic_extract_ignores_carrier_name():
    text = (
        "HCFA CMS-1500\n"
        "Carrier Name: American Family Insurance\n"
        "Claim Number: CLM-2024-551122\n"
        "Date of Birth: 03/14/1988\n"
    )
    fields = heuristic_extract(text, ["name", "claim_id", "dob"])
    assert "name" not in fields
    assert fields["claim_id"][0] == "CLM-2024-551122"


def test_heuristic_extract_patient_name_still_works():
    text = (
        "HCFA\n"
        "Carrier Name: American Family Insurance\n"
        "Patient Name: Jane Q Public\n"
        "Claim Number: CLM-9\n"
    )
    fields = heuristic_extract(text, ["name", "claim_id"])
    assert "Jane" in fields["name"][0]
    assert "American Family" not in fields["name"][0]
