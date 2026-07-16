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
