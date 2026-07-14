"""Tests for the chronological chained analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.batch_runner import run_batch
from src.pipeline.markdown_convert import (
    convert_to_markdown,
    pdf_to_markdown,
    png_to_markdown,
    text_to_structured_markdown,
)
from src.pipeline.orchestrator import DocumentAnalysisOrchestrator, analyze_document
from src.pipeline.stages import (
    ClassifyStage,
    ExtractStage,
    MarkdownConvertStage,
    SummarizeStage,
    VisionLLMStage,
)
from src.pipeline.types import AnalysisDocument, AnalysisContext, StageResult
from src.utils.config import Config
from src.utils.io import load_jsonl


FIXTURES = Path(__file__).parent / "fixtures" / "sample_documents.jsonl"


def test_context_add_skips_failed_stage_payload():
    ctx = AnalysisContext(
        document=AnalysisDocument(record_id="x", text="Claim Number: CLM-1")
    )
    failed = StageResult(
        stage="classify",
        order=1,
        ok=False,
        confidence=0.0,
        flags=["classify_failed"],
        error="boom",
        payload={},
    )
    ctx.add(failed)
    assert ctx.classification is None
    assert "classify_failed" in ctx.flags

    ok = StageResult(
        stage="classify",
        order=1,
        ok=True,
        confidence=0.9,
        payload={"document_type": "loss_notice", "confidence": 0.9},
    )
    ctx.add(ok)
    assert ctx.classification["document_type"] == "loss_notice"


def test_text_to_structured_markdown_emits_fields_table():
    text = (
        "AUTOMOBILE LOSS NOTICE\n"
        "Claim Number: CLM-1\n"
        "Date of Loss: 2024-01-15\n"
        "Loss Type: collision\n"
        "LOSS DETAILS\n"
        "Loss Location: 123 Main St\n"
    )
    md = text_to_structured_markdown(text)
    assert md.startswith("# ")
    assert "| Field | Value |" in md
    assert "CLM-1" in md
    assert "2024-01-15" in md
    assert "## Loss Details" in md or "LOSS DETAILS" in md or "Loss Details" in md


def test_png_to_markdown_uses_fallback_text(tmp_path: Path):
    from PIL import Image

    img_path = tmp_path / "page.png"
    Image.new("RGB", (200, 80), color=(255, 255, 255)).save(img_path)
    fallback = "LOSS NOTICE\nClaim Number: CLM-9\nDate of Loss: 2023-02-17\n"
    result = png_to_markdown(img_path, fallback_text=fallback)
    assert result.source_kind == "png"
    assert "CLM-9" in result.markdown
    assert "| Field | Value |" in result.markdown
    assert result.approx_tokens > 0


def test_pdf_to_markdown_extracts_text(tmp_path: Path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "loss.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(72, 720, "AUTOMOBILE LOSS NOTICE")
    c.drawString(72, 700, "Claim Number: CLM-PDF-1")
    c.drawString(72, 680, "Date of Loss: 2022-11-01")
    c.drawString(72, 660, "Loss Type: fire")
    c.save()

    result = pdf_to_markdown(pdf_path)
    assert result.source_kind == "pdf"
    assert result.pages >= 1
    assert "CLM-PDF-1" in result.markdown or "CLM-PDF-1" in result.plain_text
    assert result.approx_tokens < 1600  # cheaper than a vision-token page proxy


def test_convert_priority_prefers_pdf_over_text(tmp_path: Path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "doc.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(72, 720, "Claim Number: FROM-PDF")
    c.save()
    result = convert_to_markdown(text="Claim Number: FROM-TEXT", pdf_path=pdf_path)
    assert result.source_kind == "pdf"
    assert "FROM-PDF" in result.markdown or "FROM-PDF" in result.plain_text


def test_analyze_single_chains_stages_in_order():
    cfg = Config.load()
    text = load_jsonl(FIXTURES)[1]["text"]  # loss_notice sample
    result = analyze_document(
        text,
        record_id="test-loss",
        claim_id="CLM-TEST",
        cfg=cfg,
        enable_vision=True,
    )
    stage_names = [s["stage"] for s in result["stages"]]
    assert stage_names == [
        "to_markdown",
        "classify",
        "extract",
        "vision_llm",
        "summarize",
    ]
    assert [s["order"] for s in result["stages"]] == [0, 1, 2, 3, 4]
    assert all(s["ok"] for s in result["stages"])
    assert result["markdown"]["markdown"]
    assert "| Field | Value |" in result["markdown"]["markdown"]
    assert result["classification"]["document_type"] == "loss_notice"
    assert result["extraction"]["fields_flat"].get("date_of_loss")
    assert result["memo"] and "ADJUSTER MEMO" in result["memo"]
    # Summarize must react to prior stages including markdown
    grounded = result["summary"]["grounded_in"]
    assert "to_markdown" in grounded
    assert "classify" in grounded
    assert "extract" in grounded
    assert "vision_llm" in grounded
    assert result["summary"]["input_from"] == "markdown"
    assert result["vision"]["llm_input_mode"] == "markdown"


def test_analyze_image_path_converts_before_llm(tmp_path: Path):
    from src.extraction.render_forms import render_page

    cfg = Config.load()
    text = load_jsonl(FIXTURES)[1]["text"]
    img, _ = render_page(text)
    img_path = tmp_path / "loss.png"
    img.save(img_path)

    result2 = analyze_document(
        text,
        record_id="img-loss-2",
        image_path=img_path,
        cfg=cfg,
        enable_vision=True,
    )
    assert result2["stages"][0]["stage"] == "to_markdown"
    assert result2["markdown"]["source_kind"] in {"png", "text"}
    assert result2["vision"]["llm_input_mode"] == "markdown"
    assert result2["summary"]["input_from"] == "markdown"
    # Token savings vs vision-page proxy should be reported when image present
    assert result2["markdown"]["token_baseline"] >= result2["markdown"]["approx_tokens"]


def test_orchestrator_preserves_custom_initiation_order():
    cfg = Config.load()
    stages = [
        MarkdownConvertStage(cfg=cfg, order=0),
        ClassifyStage(cfg=cfg, order=1),
        SummarizeStage(cfg=cfg, order=2),
        ExtractStage(cfg=cfg, order=3, render_image=False),
    ]
    orch = DocumentAnalysisOrchestrator(cfg=cfg, stages=stages)
    assert orch.stage_names == ["to_markdown", "classify", "summarize", "extract"]
    row = load_jsonl(FIXTURES)[0]
    ctx = orch.analyze(row)
    assert [s.stage for s in ctx.stages] == [
        "to_markdown",
        "classify",
        "summarize",
        "extract",
    ]


def test_vision_skipped_when_disabled():
    cfg = Config.load()
    orch = DocumentAnalysisOrchestrator(cfg=cfg, enable_vision=False)
    assert orch.stage_names == [
        "to_markdown",
        "classify",
        "extract",
        "vision_llm",
        "summarize",
    ]
    ctx = orch.analyze(AnalysisDocument.from_row(load_jsonl(FIXTURES)[0]))
    vision = ctx.prior("vision_llm")
    assert vision is not None
    assert vision.payload.get("skipped") is True
    assert "vision_llm_skipped" in vision.flags
    assert ctx.markdown is not None


def test_batch_runner_writes_review_queue(tmp_path: Path):
    cfg = Config.load()
    summary = run_batch(
        FIXTURES,
        tmp_path / "batch",
        cfg=cfg,
        enable_vision=True,
        limit=3,
    )
    assert summary["n"] == 3
    assert summary["chain"] == [
        "to_markdown",
        "classify",
        "extract",
        "vision_llm",
        "summarize",
    ]
    results = load_jsonl(tmp_path / "batch" / "batch_results.jsonl")
    assert len(results) == 3
    assert all(r.get("memo") for r in results)
    assert all((r.get("markdown") or {}).get("markdown") for r in results)
    assert (tmp_path / "batch" / "human_review_queue.jsonl").exists()
    assert (tmp_path / "batch" / "batch_summary.json").exists()


def test_extraction_reacts_to_classification():
    cfg = Config.load()
    orch = DocumentAnalysisOrchestrator(
        cfg=cfg,
        stages=[
            MarkdownConvertStage(cfg=cfg, order=0),
            ClassifyStage(cfg=cfg, order=1),
            ExtractStage(cfg=cfg, order=2, render_image=False),
            VisionLLMStage(cfg=cfg, order=3, enabled=False),
            SummarizeStage(cfg=cfg, order=4),
        ],
    )
    loss_doc = next(r for r in load_jsonl(FIXTURES) if r["document_type"] == "loss_notice")
    ctx = orch.analyze(loss_doc)
    assert ctx.classification["document_type"] == "loss_notice"
    assert ctx.extraction["document_type"] == "loss_notice"
    assert "date_of_loss" in (ctx.extraction["fields_flat"] or {})
    assert ctx.extraction.get("input_from") == "markdown"
