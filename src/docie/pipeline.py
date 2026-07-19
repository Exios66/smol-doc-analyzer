"""
End-to-end DICIE orchestrator matching Fig. 1:

  Input → Document Processing → Document Classification
        → Information Extraction → Aggregated Output / Response
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from src.docie.aggregate import aggregate_prediction, push_downstream
from src.docie.applications import list_applications, load_application
from src.docie.classify import classify_document
from src.docie.extract import extract_information
from src.docie.processing import process_document_input
from src.docie.types import DociePrediction
from src.utils.config import Config
from src.utils.io import load_jsonl, write_json, write_jsonl
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


class DociePipeline:
    """
    Paper-aligned Document Image Classification and Information Extraction
    pipeline (Fig. 1).
    """

    def __init__(
        self,
        application: str = "salvage_claims",
        *,
        cfg: Config | None = None,
        cache_dir: Path | None = None,
        dpi: int = 300,
        grayscale: bool = True,
        run_ocr: bool = True,
        vit_model_dir: Path | None = None,
        extractor_dir: Path | None = None,
        downstream_sink: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.cfg = cfg or Config.load()
        self.profile = load_application(application)
        self.cache_dir = Path(
            cache_dir or (self.cfg.pipeline_cache_dir / "docie" / self.profile.name)
        )
        self.dpi = dpi
        self.grayscale = grayscale
        self.run_ocr = run_ocr
        self.vit_model_dir = vit_model_dir
        self.extractor_dir = extractor_dir or self._default_extractor_dir()
        self.downstream_sink = downstream_sink

    def _default_extractor_dir(self) -> Path | None:
        primary = self.cfg.models_dir / "extractor"
        smoke = self.cfg.models_dir / "extractor_smoke"
        if (primary / "config.json").exists():
            return primary
        if (smoke / "config.json").exists():
            return smoke
        return None

    def process(
        self,
        *,
        record_id: str = "adhoc",
        pdf_path: str | Path | None = None,
        image_path: str | Path | None = None,
        source_path: str | Path | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DociePrediction:
        """Run the full Fig. 1 chain for one inbound document."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        processed = process_document_input(
            record_id=record_id,
            pdf_path=pdf_path,
            image_path=image_path,
            source_path=source_path,
            text=text,
            cache_dir=self.cache_dir / "pages",
            application=self.profile.name,
            dpi=self.dpi,
            grayscale=self.grayscale,
            run_ocr=self.run_ocr,
            metadata=metadata,
        )
        timings["document_processing"] = _ms(t0)
        logger.info(
            "Stage 1 processing %s — %d page(s), kind=%s",
            record_id,
            len(processed.pages),
            processed.source_kind,
        )

        t1 = time.perf_counter()
        classification = classify_document(
            processed,
            self.profile,
            vit_model_dir=self.vit_model_dir,
        )
        timings["document_classification"] = _ms(t1)
        logger.info(
            "Stage 2 classify %s → %s (%.3f)",
            record_id,
            classification.label,
            classification.confidence,
        )

        t2 = time.perf_counter()
        extraction = extract_information(
            processed,
            self.profile,
            document_type=classification.label,
            extractor_dir=self.extractor_dir,
        )
        timings["information_extraction"] = _ms(t2)
        logger.info(
            "Stage 3 extract %s — %d field(s), backend=%s",
            record_id,
            sum(1 for v in extraction.fields_flat.values() if v),
            extraction.backend,
        )

        t3 = time.perf_counter()
        prediction = aggregate_prediction(
            processed=processed,
            classification=classification,
            extraction=extraction,
            profile=self.profile,
            stage_timings_ms=timings,
        )
        timings["output_aggregation"] = _ms(t3)
        prediction.stage_timings_ms = timings

        # Fig. 1 final arrow: send response / push to downstream systems
        push_downstream(prediction, sink=self.downstream_sink)
        return prediction

    def process_row(self, row: dict[str, Any]) -> DociePrediction:
        record_id = str(row.get("record_id") or row.get("claim_id") or "unknown")
        return self.process(
            record_id=record_id,
            pdf_path=row.get("pdf_path"),
            image_path=row.get("image_path"),
            source_path=row.get("source_path") or row.get("path") or row.get("file_path"),
            text=row.get("text"),
            metadata={
                k: v
                for k, v in row.items()
                if k
                not in {
                    "record_id",
                    "claim_id",
                    "pdf_path",
                    "image_path",
                    "source_path",
                    "path",
                    "file_path",
                    "text",
                }
            },
        )

    def process_many(
        self, rows: Sequence[dict[str, Any]]
    ) -> list[DociePrediction]:
        return [self.process_row(row) for row in rows]


def process_document(
    *,
    application: str = "salvage_claims",
    record_id: str = "adhoc",
    pdf_path: str | Path | None = None,
    image_path: str | Path | None = None,
    source_path: str | Path | None = None,
    text: str | None = None,
    cfg: Config | None = None,
    run_ocr: bool = True,
) -> dict[str, Any]:
    """Convenience API: one document → Fig. 1 prediction dict."""
    pipe = DociePipeline(application=application, cfg=cfg, run_ocr=run_ocr)
    return pipe.process(
        record_id=record_id,
        pdf_path=pdf_path,
        image_path=image_path,
        source_path=source_path,
        text=text,
    ).to_dict()


def run_file(
    inp: Path,
    out: Path,
    *,
    application: str = "salvage_claims",
    cfg: Config | None = None,
    limit: int | None = None,
    run_ocr: bool = True,
) -> Path:
    cfg = cfg or Config.load()
    rows = load_jsonl(inp)
    if limit is not None:
        rows = rows[:limit]
    pipe = DociePipeline(application=application, cfg=cfg, run_ocr=run_ocr)
    predictions = [pipe.process_row(row) for row in rows]
    results = [p.to_dict() for p in predictions]
    write_jsonl(out, results)

    summary = {
        "n": len(results),
        "application": application,
        "chain": [
            "document_processing",
            "document_classification",
            "information_extraction",
            "output_aggregation",
        ],
        "needs_human_review": sum(1 for p in predictions if p.needs_human_review),
        "label_counts": _count_labels(predictions),
        "avg_classification_confidence": _avg(
            [p.classification.confidence for p in predictions]
        ),
        "avg_extraction_confidence": _avg(
            [p.extraction.confidence for p in predictions]
        ),
    }
    write_json(out.with_suffix(".summary.json"), summary)
    # Also write a human-review queue for low-confidence / missing fields
    review_path = out.with_name(out.stem + ".human_review.jsonl")
    write_jsonl(
        review_path,
        [p.response_payload() for p in predictions if p.needs_human_review],
    )
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"docie-{application}-{out.stem}",
            stage="docie_pipeline",
            source=str(inp),
            prompt_version="docie_fig1_v1",
            model="→".join(summary["chain"]),
            extra=summary,
        ),
    )
    return out


def _count_labels(predictions: Sequence[DociePrediction]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in predictions:
        counts[p.classification.label] = counts.get(p.classification.label, 0) + 1
    return counts


def _avg(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper Fig. 1 DICIE pipeline: "
            "document processing → classification → information extraction → output"
        )
    )
    parser.add_argument(
        "--application",
        "-a",
        default="salvage_claims",
        choices=list_applications(),
        help="Insurance application profile (medical_bills | salvage_claims | acord)",
    )
    parser.add_argument("--in", dest="inp", type=Path, help="Input JSONL of documents")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--text", type=str, default=None, help="Analyze a single text blob")
    parser.add_argument("--image", type=Path, default=None, help="Document page image")
    parser.add_argument("--pdf", type=Path, default=None, help="Multi/single-page PDF")
    parser.add_argument("--record-id", type=str, default="adhoc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-ocr", action="store_true", help="Skip pytesseract OCR")
    parser.add_argument(
        "--response-only",
        action="store_true",
        help="Print compact downstream response payload instead of full prediction",
    )
    parser.add_argument("--vit-model-dir", type=Path, default=None)
    parser.add_argument("--extractor-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = Config.load()

    pipe = DociePipeline(
        application=args.application,
        cfg=cfg,
        dpi=args.dpi,
        run_ocr=not args.no_ocr,
        vit_model_dir=args.vit_model_dir,
        extractor_dir=args.extractor_dir,
    )

    if args.text is not None or args.image or args.pdf:
        prediction = pipe.process(
            record_id=args.record_id,
            text=args.text,
            image_path=args.image,
            pdf_path=args.pdf,
        )
        payload = (
            prediction.response_payload()
            if args.response_only
            else prediction.to_dict()
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if not args.inp:
        parser.error("Provide --in JSONL, or --text / --image / --pdf")

    out = args.out or (
        cfg.pipeline_output_dir / "docie" / f"{args.application}_{args.inp.stem}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    path = run_file(
        args.inp,
        out,
        application=args.application,
        cfg=cfg,
        limit=args.limit,
        run_ocr=not args.no_ocr,
    )
    print(path)


if __name__ == "__main__":
    main()
