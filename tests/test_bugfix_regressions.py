"""Regression tests for bugs fixed in the full audit."""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import pytest
from PIL import Image

from src.extraction.render_forms import label_words, render_page
from src.generation.noise_injection import garble_text
from src.generation.skeleton_sampler import _random_date, sample_batch, write_splits
from src.pipeline.markdown_convert import pdf_to_markdown, png_to_markdown
from src.pipeline.stages import (
    MarkdownConvertStage,
    _cache_safe_id,
    _heuristic_classify,
    _prefer_model_dir,
)
from src.pipeline.types import AnalysisContext, AnalysisDocument
from src.utils.config import Config
from src.utils.io import read_json


def test_png_ocr_failure_preserves_backend_and_low_confidence(tmp_path: Path, monkeypatch):
    img_path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img_path)

    monkeypatch.setattr(
        "src.pipeline.markdown_convert._ocr_image",
        lambda path: ("", "pytesseract_failed"),
    )
    conversion = png_to_markdown(img_path)
    assert conversion.backend == "pytesseract_failed"
    assert conversion.plain_text == ""

    cfg = Config.load()
    stage = MarkdownConvertStage(cfg=cfg)
    ctx = AnalysisContext(
        document=AnalysisDocument(record_id="ocr-fail", text="", image_path=img_path)
    )
    result = stage.run(ctx)
    assert result.ok
    assert result.confidence <= 0.35
    assert "low_confidence_markdown" in result.flags


def test_pdf_empty_text_marks_backend_empty(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "empty.pdf"
    # Minimal valid-ish PDF bytes are not required; stub extractors.
    pdf_path.write_bytes(b"%PDF-1.4\n%\n")

    monkeypatch.setattr(
        "src.pipeline.markdown_convert._pdf_extract_pymupdf",
        lambda path: ("", 1, "pymupdf"),
    )
    conversion = pdf_to_markdown(pdf_path)
    assert conversion.backend.endswith("_empty") or conversion.backend in {"empty", "failed"}
    assert conversion.plain_text == ""


def test_acord_form_substring_does_not_false_positive_loss_notice():
    labels = [
        "application_commercial",
        "certificate_evidence",
        "loss_notice",
        "policy_change_endorsement",
    ]
    text = "COMMERCIAL INSURANCE APPLICATION\nACORD Form: 125\nCoverage sections"
    label, _ = _heuristic_classify(text, labels)
    assert label == "application_commercial"

    text2 = "CERTIFICATE OF INSURANCE\nACORD Form: 25\nEvidence of insurance"
    label2, _ = _heuristic_classify(text2, labels)
    assert label2 == "certificate_evidence"


def test_cache_safe_id_avoids_slash_underscore_collision():
    a = _cache_safe_id("foo/bar")
    b = _cache_safe_id("foo_bar")
    assert a != b


def test_prefer_model_dir_chooses_production_over_smoke(tmp_path: Path):
    models = tmp_path / "models"
    smoke = models / "classifier_smoke"
    prod = models / "classifier"
    smoke.mkdir(parents=True)
    prod.mkdir(parents=True)
    (smoke / "config.json").write_text("{}", encoding="utf-8")
    (prod / "config.json").write_text("{}", encoding="utf-8")
    assert _prefer_model_dir(models, "classifier", "classifier_smoke") == prod


def test_from_row_tolerates_null_word_text():
    doc = AnalysisDocument.from_row(
        {"record_id": "r1", "words": [{"text": None}, {"text": "Hello"}]}
    )
    assert "Hello" in doc.text


def test_noise_insertion_does_not_always_substitute():
    profile = {
        "char_substitution_rate": 0.0,
        "char_deletion_rate": 0.0,
        "char_insertion_rate": 1.0,
        "common_ocr_artifacts": ["~"],
        "letter_confusion": {},
        "line_break_noise_rate": 0.0,
        "word_merge_rate": 0.0,
        "word_split_rate": 0.0,
    }
    # Exclusive insert band doubles length (artifact_or_char + original); never
    # drops into the substitution branch when sub_rate is 0.
    out = garble_text("ABC", profile, random.Random(0))
    assert len(out) == 6
    assert "A" in out and "B" in out and "C" in out
    assert set(out) <= {"A", "B", "C", "~"}

    # When only substitution is active, length stays the same (no insert overlap).
    sub_only = {
        **profile,
        "char_insertion_rate": 0.0,
        "char_substitution_rate": 1.0,
    }
    out2 = garble_text("ABC", sub_only, random.Random(1))
    assert len(out2) == 3



def test_random_date_respects_end_inclusive_and_loss_after_effective():
    rng = random.Random(0)
    d = _random_date(rng, date(2024, 1, 1), date(2024, 1, 1))
    assert d == "2024-01-01"

    cfg = Config.load()
    dist = read_json(cfg.profiles_dir / "insurance_distributions.json")
    schema = read_json(cfg.claim_schema_path)
    skeletons = sample_batch(n=40, seed=7, dist=dist, schema=schema)
    for sk in skeletons:
        assert sk["loss_event"]["date_of_loss"] >= sk["policy"]["effective_date"]


def test_write_splits_keeps_nonempty_train(tmp_path: Path):
    skeletons = [
        {
            "claim_id": f"CLM-{i}",
            "document_type": "loss_notice",
            "policy": {},
            "loss_event": {},
            "parties": {},
            "financials": {},
        }
        for i in range(2)
    ]
    splits = write_splits(skeletons, tmp_path / "splits.json", seed=0)
    assert len(splits["train"]) >= 1


def test_label_words_stops_before_acord_header():
    text = (
        "Claim Number: CLM-2024-000001\n"
        "ACORD Form: 101\n"
        "POLICY CHANGE\n"
        "Named Insured: Avery Nguyen\n"
    )
    _img, words, _trunc = render_page(text)
    labeled = label_words(words)
    claim_tokens = [w for w in labeled if w["label"].endswith("claim_id")]
    claim_text = " ".join(w["text"] for w in claim_tokens)
    assert "CLM-2024-000001" in claim_text
    assert "ACORD" not in claim_text
    assert "Form:" not in claim_text
    assert "101" not in claim_text


def test_vision_copy_on_write_preserves_extract_stage_payload():
    from src.pipeline.stages import VisionLLMStage

    cfg = Config.load()
    original_fields = {"claim_id": ["CLM-1"], "date_of_loss": ["2024-01-15"]}
    original_flat = {"claim_id": "CLM-1", "date_of_loss": "2024-01-15"}
    extract_payload = {
        "fields": original_fields,
        "fields_flat": original_flat,
        "backend": "heuristic",
    }
    ctx = AnalysisContext(
        document=AnalysisDocument(
            record_id="v-cow",
            text="LOSS NOTICE\nClaim Number: CLM-1\nDate of Loss: 2024-01-15\n",
        )
    )
    from src.pipeline.types import StageResult

    ctx.add(
        StageResult(
            stage="extract",
            order=2,
            ok=True,
            confidence=0.7,
            payload=extract_payload,
        )
    )
    stage = VisionLLMStage(cfg=cfg, enabled=True)
    result = stage.run(ctx)
    assert result.ok
    # Historical extract stage payload must remain unchanged.
    assert ctx.prior("extract").payload["fields_flat"] == original_flat
    # Live extraction slot may be refined via a new dict.
    assert ctx.extraction is not extract_payload
    assert "CLM-1" in str(ctx.extraction.get("fields_flat", {}))


def test_load_eval_set_rejects_negative_n_samples():
    from evaluation.eval_harness import load_eval_set
    from src.utils.config import REPO_ROOT

    path = REPO_ROOT / "data" / "eval" / "eval_set.jsonl"
    with pytest.raises(ValueError):
        load_eval_set(path, n_samples=-1)


def test_annotate_updates_correct_when_judge_score_present():
    from evaluation.metrics import annotate_records

    rows = [
        {
            "task": "memo_generation",
            "prediction": "short memo without required sections",
            "ground_truth": "ref",
            "error": None,
            "judge_score": 2.0,
        }
    ]
    annotate_records(rows)
    assert rows[0]["score"] == pytest.approx(0.4)
    assert rows[0]["correct"] is False
    assert rows[0]["score_source"] == "judge"


# ---------------------------------------------------------------------------
# Round-2 audit regressions (issues #28–#35)
# ---------------------------------------------------------------------------


def test_analyze_impl_accepts_local_path_without_url_download(tmp_path: Path, monkeypatch):
    """#28: slash attachments must use local_path, not file_url."""
    import asyncio

    from src.discord_bot.tools import analyze_insurance_document_impl

    # Use a .txt inbox path so the pipeline does not need a real PDF parser.
    txt = tmp_path / "claim.txt"
    txt.write_text(
        "AUTOMOBILE LOSS NOTICE\nClaim Number: CLM-LOCAL-1\nDate of Loss: 2024-01-15\n",
        encoding="utf-8",
    )

    async def _boom(*_a, **_k):
        raise AssertionError("file_url download must not run for local_path")

    monkeypatch.setattr("src.discord_bot.tools._download_url", _boom)
    out = asyncio.run(
        analyze_insurance_document_impl(
            local_path=txt,
            enable_vision=False,
            record_id="test-local-path",
        )
    )
    assert out.get("ok") is True
    assert out["analysis"]["record_id"] == "test-local-path"


def test_download_url_rejects_redirect_to_private_ip(monkeypatch):
    """#29: redirect hops must be re-validated (SSRF)."""
    import asyncio
    import io
    import urllib.error

    from src.discord_bot.tools import _download_url

    class _FakeResp(io.BytesIO):
        def __init__(self, code, headers, body=b""):
            super().__init__(body)
            self.code = code
            self.headers = headers
            self.status = code

        def getcode(self):
            return self.code

    calls = {"n": 0}

    def fake_open(req, timeout=60):  # noqa: ARG001
        calls["n"] += 1
        url = req.full_url
        if "public.example" in url:
            raise urllib.error.HTTPError(
                url,
                302,
                "Found",
                hdrs={"Location": "http://127.0.0.1/secret"},
                fp=io.BytesIO(b""),
            )
        raise AssertionError(f"unexpected open: {url}")

    class _FakeOpener:
        def open(self, req, timeout=60):
            return fake_open(req, timeout=timeout)

    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *_a, **_k: _FakeOpener(),
    )
    # Host validation for public.example — stub DNS to a public IP.
    def fake_getaddrinfo(host, *a, **k):  # noqa: ARG001
        if host == "public.example":
            return [(0, 0, 0, "", ("93.184.216.34", 0))]
        if host in {"127.0.0.1", "localhost"}:
            return [(0, 0, 0, "", ("127.0.0.1", 0))]
        raise OSError(f"blocked resolve for {host}")

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)

    async def _run():
        with pytest.raises(ValueError, match="private|blocked|loopback|127"):
            await _download_url("http://public.example/doc.pdf", Path("/tmp/ssrf-test.bin"))

    asyncio.run(_run())
    assert calls["n"] == 1


def test_docie_upload_path_rejects_traversal(tmp_path: Path):
    """#30: record_id must not escape the upload temp directory."""
    from src.docie.serve import _safe_upload_path, _sanitize_record_id

    assert ".." not in _sanitize_record_id("../../../tmp/escape")
    assert "/" not in _sanitize_record_id("a/b/c")
    path = _safe_upload_path(tmp_path, "../../../tmp/escape", ".pdf")
    assert path.parent == tmp_path.resolve()
    assert path.name.endswith(".pdf")
    assert ".." not in path.name


def test_upsert_replaces_omitted_ground_truth_fields(tmp_path: Path):
    """#31: corrective upsert must drop stale gold fields."""
    from src.storage.store import DocumentStore
    from src.storage.types import DocumentRecord, FieldRecord

    store = DocumentStore(tmp_path / "docs.db")
    store.upsert_document(
        DocumentRecord(
            document_id="doc-1",
            application="medical_bills",
            document_type="hcfa",
            text="Patient Name: Ada",
            fields=[
                FieldRecord("name", "Ada"),
                FieldRecord("claim_id", "CLM-1"),
                FieldRecord("dob", "1990-01-01"),
            ],
        )
    )
    store.upsert_document(
        DocumentRecord(
            document_id="doc-1",
            application="medical_bills",
            document_type="hcfa",
            text="Patient Name: Ada",
            fields=[
                FieldRecord("name", "Ada Lovelace"),
                FieldRecord("claim_id", "CLM-1"),
            ],
        )
    )
    fields = store.get_document("doc-1").ground_truth_fields()
    assert fields["name"] == "Ada Lovelace"
    assert fields["claim_id"] == "CLM-1"
    assert "dob" not in fields


def test_assign_split_keeps_record_id_surfaces_together(tmp_path: Path):
    """#32: typed + noisy rows sharing record_id stay in one split."""
    import pandas as pd

    from src.classification.random_forest import (
        SURFACE_HANDWRITING_OCR,
        SURFACE_TYPED,
        assign_split_column,
    )

    rows = []
    for i in range(12):
        rid = f"r{i}"
        label = "loss_notice" if i % 2 == 0 else "repair_estimate"
        rows.append(
            {
                "record_id": rid,
                "document_type": label,
                "text": f"typed {rid}",
                "surface": SURFACE_TYPED,
            }
        )
        rows.append(
            {
                "record_id": rid,
                "document_type": label,
                "text": f"noisy {rid}",
                "surface": SURFACE_HANDWRITING_OCR,
            }
        )
    frame = assign_split_column(pd.DataFrame(rows), splits_path=tmp_path / "missing.json")
    grouped = frame.groupby("record_id")["split"].nunique()
    assert (grouped == 1).all()

    # Misaligned splits.json (no matching IDs) must not crash.
    bad = tmp_path / "splits.json"
    bad.write_text('{"train":["other-1"],"val":[],"test":["other-2"]}', encoding="utf-8")
    frame2 = assign_split_column(pd.DataFrame(rows), splits_path=bad)
    assert set(frame2["split"]) <= {"train", "val", "test"}
    assert (frame2.groupby("record_id")["split"].nunique() == 1).all()


def test_carrier_name_not_extracted_as_patient_name():
    """#33: Carrier Name must not become patient name."""
    from src.docie.extract import heuristic_extract

    carrier_only = "CLAIM FORM\nCarrier Name: American Family\nClaim Number: CLM-9\n"
    fields = heuristic_extract(carrier_only, ["name", "claim_id"])
    assert "name" not in fields
    assert fields["claim_id"][0] == "CLM-9"

    both = (
        "HCFA\nPatient Name: Jane Q Public\nCarrier Name: American Family\n"
        "Claim Number: CLM-10\n"
    )
    fields2 = heuristic_extract(both, ["name", "claim_id"])
    assert "Jane" in fields2["name"][0]
    assert "American Family" not in fields2["name"]


def test_render_forms_safe_ids_avoid_collision(tmp_path: Path):
    """#34: :: vs __ record IDs must not overwrite the same PNG."""
    from src.extraction.render_forms import _cache_safe_id, render_documents
    from src.utils.io import write_jsonl

    a = "CLM-1::loss_notice::0"
    b = "CLM-1__loss_notice__0"
    assert _cache_safe_id(a) != _cache_safe_id(b)
    docs = tmp_path / "docs.jsonl"
    write_jsonl(
        docs,
        [
            {
                "record_id": a,
                "claim_id": "CLM-1",
                "document_type": "loss_notice",
                "text": "LOSS NOTICE A\nClaim Number: CLM-1\n",
            },
            {
                "record_id": b,
                "claim_id": "CLM-1",
                "document_type": "loss_notice",
                "text": "LOSS NOTICE B\nClaim Number: CLM-1\n",
            },
        ],
    )
    out = render_documents(docs, tmp_path / "rendered")
    images = list((tmp_path / "rendered" / "images").glob("*.png"))
    assert len(images) == 2


def test_eval_harness_zero_cost_on_free_model_fallback():
    """#35: free OpenRouter models must record $0 and actual model_id."""
    from evaluation.eval_harness import FrontierBackend, ModelPricing, run_eval

    class FreeFallbackBackend(FrontierBackend):
        def __init__(self):
            self.name = "anthropic"
            self.model_slug = "anthropic/claude-sonnet-4.5"
            self.pricing = ModelPricing(input_per_million=3.0, output_per_million=15.0)
            self.client = None  # unused — run() is overridden

        def run(self, task, example):
            return "loss_notice", 1000, 100, 0.2, "openrouter/free"

    results = run_eval(
        tasks=["classification"],
        backends={"anthropic": FreeFallbackBackend()},
        eval_set=[
            {
                "example_id": "cls-free",
                "task": "classification",
                "prompt_fields": {"document_text": "x"},
                "ground_truth": "loss_notice",
            }
        ],
        frontier_pricing={"anthropic": ModelPricing(3.0, 15.0)},
        run_id="free-test",
        dry_run=False,
    )
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].model_id == "openrouter/free"
    assert results[0].cost_usd == 0.0
