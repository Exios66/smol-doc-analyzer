"""
Output stage — aggregate classification + extraction (paper Fig. 1 / §VI).

Applies application business rules, decides human-review routing, and
shapes the response payload for downstream systems.
"""

from __future__ import annotations

from typing import Any, Callable

from src.docie.applications import ApplicationProfile
from src.docie.types import (
    ClassificationResult,
    DociePrediction,
    ExtractionResult,
    ProcessedDocument,
)


def aggregate_prediction(
    *,
    processed: ProcessedDocument,
    classification: ClassificationResult,
    extraction: ExtractionResult,
    profile: ApplicationProfile,
    stage_timings_ms: dict[str, float] | None = None,
) -> DociePrediction:
    """Combine Stage 2 + Stage 3 into the final Fig. 1 prediction."""
    flags: list[str] = []
    flags.extend(classification.flags)
    flags.extend(extraction.flags)

    if processed.source_kind == "empty" or not processed.pages:
        flags.append("empty_document")

    # OCR quality signal — empty OCR on image/pdf inputs is a review trigger
    if processed.source_kind in {"pdf", "image"}:
        empty_ocr_pages = sum(1 for p in processed.pages if not (p.text or "").strip())
        if empty_ocr_pages:
            flags.append("ocr_empty_pages")
        weak_backends = {"none", "pytesseract_failed"}
        if all(p.ocr_backend in weak_backends for p in processed.pages):
            flags.append("ocr_unavailable")

    needs_review = any(
        f.startswith("low_confidence")
        or f.startswith("missing_expected_field")
        or f in {"empty_document", "ocr_empty_pages", "ocr_unavailable", "no_pages"}
        for f in flags
    )
    if (
        classification.confidence < profile.review_confidence_threshold
        or extraction.confidence < profile.review_confidence_threshold
    ):
        needs_review = True
        if "human_review_threshold" not in flags:
            flags.append("human_review_threshold")

    processing = {
        "source_kind": processed.source_kind,
        "source_path": processed.source_path,
        "n_pages": len(processed.pages),
        "dpi": processed.pages[0].dpi if processed.pages else None,
        "grayscale": processed.pages[0].grayscale if processed.pages else None,
        "ocr_backends": sorted({p.ocr_backend for p in processed.pages}),
        "page_image_paths": [str(p.image_path) for p in processed.pages],
    }

    return DociePrediction(
        record_id=processed.record_id,
        application=profile.name,
        classification=classification,
        extraction=extraction,
        processing=processing,
        flags=list(dict.fromkeys(flags)),
        needs_human_review=needs_review,
        stage_timings_ms=dict(stage_timings_ms or {}),
    )


def push_downstream(
    prediction: DociePrediction,
    *,
    sink: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """
    Send / return the aggregated response for downstream workflows.

    When ``sink`` is provided it receives the compact response payload
    (e.g. webhook, queue publisher, claim-center updater). Always returns
    the payload for the FastAPI / CLI caller.
    """
    payload = prediction.response_payload()
    if sink is not None:
        sink(payload)
    return payload
