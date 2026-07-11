"""
Chronological document-analysis orchestrator.

One analyze action chains every initiated stage in initiation order:
  classify → extract → vision_llm → summarize

Each stage receives the accumulating AnalysisContext and may react to
prior stage payloads (document type, extracted fields, vision refinements).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from src.pipeline.stages import (
    ClassifyStage,
    ExtractStage,
    PipelineStage,
    SummarizeStage,
    VisionLLMStage,
)
from src.pipeline.types import AnalysisContext, AnalysisDocument
from src.utils.config import Config
from src.utils.io import load_jsonl, write_json, write_jsonl
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)


class DocumentAnalysisOrchestrator:
    """
    Single-action orchestrator: stages are registered in initiation order and
    executed chronologically. Later stages always see earlier StageResults.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        stages: Sequence[PipelineStage] | None = None,
        classifier_dir: Path | None = None,
        extractor_dir: Path | None = None,
        enable_vision: bool | None = None,
    ) -> None:
        self.cfg = cfg or Config.load()
        if stages is not None:
            self.stages = list(stages)
        else:
            vision_enabled = (
                self.cfg.vision_llm_enabled if enable_vision is None else enable_vision
            )
            self.stages = [
                ClassifyStage(cfg=self.cfg, model_dir=classifier_dir, order=0),
                ExtractStage(cfg=self.cfg, model_dir=extractor_dir, order=1),
                VisionLLMStage(cfg=self.cfg, order=2, enabled=vision_enabled),
                SummarizeStage(cfg=self.cfg, order=3),
            ]
        # Preserve initiation order; do not re-sort by name.
        self.stages.sort(key=lambda s: s.order)

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def analyze(self, document: AnalysisDocument | dict[str, Any]) -> AnalysisContext:
        """Run the full chained analysis for one document."""
        doc = (
            document
            if isinstance(document, AnalysisDocument)
            else AnalysisDocument.from_row(document)
        )
        ctx = AnalysisContext(document=doc)
        logger.info(
            "analyze %s — chain: %s",
            doc.record_id,
            " → ".join(self.stage_names),
        )
        for stage in self.stages:
            result = stage.run(ctx)
            ctx.add(result)
            logger.info(
                "  [%d] %s ok=%s conf=%.3f flags=%s",
                result.order,
                result.stage,
                result.ok,
                result.confidence,
                result.flags,
            )
            # Soft-continue on failure so downstream stages can still flag review
            if not result.ok:
                ctx.flags.append(f"upstream_failed:{result.stage}")
        return ctx

    def analyze_many(
        self, documents: Sequence[AnalysisDocument | dict[str, Any]]
    ) -> list[AnalysisContext]:
        return [self.analyze(doc) for doc in documents]


def analyze_document(
    text: str,
    *,
    record_id: str = "adhoc",
    claim_id: str | None = None,
    image_path: str | Path | None = None,
    cfg: Config | None = None,
    enable_vision: bool | None = None,
) -> dict[str, Any]:
    """Convenience API: one document → full chained analysis dict."""
    orch = DocumentAnalysisOrchestrator(cfg=cfg, enable_vision=enable_vision)
    ctx = orch.analyze(
        AnalysisDocument(
            record_id=record_id,
            text=text,
            claim_id=claim_id,
            image_path=image_path,
        )
    )
    return ctx.to_dict()


def run_file(
    inp: Path,
    out: Path,
    cfg: Config | None = None,
    limit: int | None = None,
    enable_vision: bool | None = None,
    classifier_dir: Path | None = None,
    extractor_dir: Path | None = None,
) -> Path:
    """Analyze a JSONL of documents and write chained results."""
    cfg = cfg or Config.load()
    rows = load_jsonl(inp)
    if limit is not None:
        rows = rows[:limit]
    orch = DocumentAnalysisOrchestrator(
        cfg=cfg,
        classifier_dir=classifier_dir,
        extractor_dir=extractor_dir,
        enable_vision=enable_vision,
    )
    results = [orch.analyze(row).to_dict() for row in rows]
    write_jsonl(out, results)
    summary = {
        "n": len(results),
        "chain": orch.stage_names,
        "low_confidence": sum(1 for r in results if r.get("low_confidence")),
        "flagged": sum(1 for r in results if r.get("flags")),
    }
    write_json(out.with_suffix(".summary.json"), summary)
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"pipeline-analyze-{out.stem}",
            stage="pipeline_analyze",
            source=str(inp),
            prompt_version="pipeline_v1",
            model="→".join(orch.stage_names),
            extra=summary,
        ),
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Run the full document-analysis chain in one action: "
            "classify → extract → vision_llm → summarize"
        )
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=Path,
        help="Input JSONL of documents (record_id, text, optional image_path)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL for chained analysis results",
    )
    parser.add_argument("--text", type=str, default=None, help="Analyze a single text blob")
    parser.add_argument("--record-id", type=str, default="adhoc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--classifier-dir", type=Path, default=None)
    parser.add_argument("--extractor-dir", type=Path, default=None)
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable Vision LLM stage (local VLM or heuristic refine)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable Vision LLM stage even if configured in env",
    )
    args = parser.parse_args()
    cfg = Config.load()

    enable_vision: bool | None
    if args.no_vision:
        enable_vision = False
    elif args.vision:
        enable_vision = True
    else:
        enable_vision = None

    if args.text:
        result = analyze_document(
            args.text,
            record_id=args.record_id,
            cfg=cfg,
            enable_vision=enable_vision if enable_vision is not None else True,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not args.inp:
        parser.error("Provide --in JSONL or --text")

    out = args.out or (cfg.pipeline_output_dir / f"analysis_{args.inp.stem}.jsonl")
    path = run_file(
        args.inp,
        out,
        cfg=cfg,
        limit=args.limit,
        enable_vision=enable_vision,
        classifier_dir=args.classifier_dir,
        extractor_dir=args.extractor_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
