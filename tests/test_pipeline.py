"""Tests for the chronological chained analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.batch_runner import run_batch
from src.pipeline.orchestrator import DocumentAnalysisOrchestrator, analyze_document
from src.pipeline.stages import ClassifyStage, ExtractStage, SummarizeStage, VisionLLMStage
from src.pipeline.types import AnalysisDocument
from src.utils.config import Config
from src.utils.io import load_jsonl


FIXTURES = Path(__file__).parent / "fixtures" / "sample_documents.jsonl"


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
    assert stage_names == ["classify", "extract", "vision_llm", "summarize"]
    # Chronological order field must match initiation order
    assert [s["order"] for s in result["stages"]] == [0, 1, 2, 3]
    assert all(s["ok"] for s in result["stages"])
    assert result["classification"]["document_type"] == "loss_notice"
    assert result["extraction"]["fields_flat"].get("date_of_loss")
    assert result["memo"] and "ADJUSTER MEMO" in result["memo"]
    # Summarize must react to prior stages
    grounded = result["summary"]["grounded_in"]
    assert "classify" in grounded
    assert "extract" in grounded
    assert "vision_llm" in grounded


def test_orchestrator_preserves_custom_initiation_order():
    cfg = Config.load()
    # Intentionally register summarize before extract to prove initiation order wins
    stages = [
        ClassifyStage(cfg=cfg, order=0),
        SummarizeStage(cfg=cfg, order=1),
        ExtractStage(cfg=cfg, order=2, render_image=False),
    ]
    orch = DocumentAnalysisOrchestrator(cfg=cfg, stages=stages)
    assert orch.stage_names == ["classify", "summarize", "extract"]
    row = load_jsonl(FIXTURES)[0]
    ctx = orch.analyze(row)
    assert [s.stage for s in ctx.stages] == ["classify", "summarize", "extract"]


def test_vision_skipped_when_disabled():
    cfg = Config.load()
    orch = DocumentAnalysisOrchestrator(cfg=cfg, enable_vision=False)
    assert orch.stage_names == ["classify", "extract", "vision_llm", "summarize"]
    ctx = orch.analyze(AnalysisDocument.from_row(load_jsonl(FIXTURES)[0]))
    vision = ctx.prior("vision_llm")
    assert vision is not None
    assert vision.payload.get("skipped") is True
    assert "vision_llm_skipped" in vision.flags


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
    assert summary["chain"] == ["classify", "extract", "vision_llm", "summarize"]
    results = load_jsonl(tmp_path / "batch" / "batch_results.jsonl")
    assert len(results) == 3
    assert all(r.get("memo") for r in results)
    assert (tmp_path / "batch" / "human_review_queue.jsonl").exists()
    assert (tmp_path / "batch" / "batch_summary.json").exists()


def test_extraction_reacts_to_classification():
    cfg = Config.load()
    orch = DocumentAnalysisOrchestrator(
        cfg=cfg,
        stages=[
            ClassifyStage(cfg=cfg, order=0),
            ExtractStage(cfg=cfg, order=1, render_image=False),
            VisionLLMStage(cfg=cfg, order=2, enabled=False),
            SummarizeStage(cfg=cfg, order=3),
        ],
    )
    loss_doc = next(r for r in load_jsonl(FIXTURES) if r["document_type"] == "loss_notice")
    ctx = orch.analyze(loss_doc)
    assert ctx.classification["document_type"] == "loss_notice"
    assert ctx.extraction["document_type"] == "loss_notice"
    assert "date_of_loss" in (ctx.extraction["fields_flat"] or {})
