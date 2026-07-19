"""Shared types for the DICIE (Fig. 1) pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OcrWord:
    """A single OCR token with optional layout coordinates (0–1000 normalized)."""

    text: str
    bbox: list[int] | None = None  # [x0, y0, x1, y1]
    conf: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "conf": self.conf,
        }


@dataclass
class PageImage:
    """One processed page produced by Stage 1 (Document Processing)."""

    page_index: int
    image_path: Path
    width: int
    height: int
    dpi: int
    grayscale: bool
    text: str = ""
    words: list[OcrWord] = field(default_factory=list)
    ocr_backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "grayscale": self.grayscale,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "ocr_backend": self.ocr_backend,
            "n_words": len(self.words),
        }


@dataclass
class ProcessedDocument:
    """Stage 1 output: page images + OCR ready for classification / extraction."""

    record_id: str
    source_path: str | None
    source_kind: str  # pdf | image | text | empty
    pages: list[PageImage] = field(default_factory=list)
    application: str = "salvage_claims"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "application": self.application,
            "n_pages": len(self.pages),
            "pages": [p.to_dict() for p in self.pages],
            "full_text": self.full_text,
            "metadata": self.metadata,
        }


@dataclass
class PageClassification:
    """Per-page classification prediction (Stage 2)."""

    page_index: int
    label: str
    confidence: float
    backend: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "label": self.label,
            "confidence": self.confidence,
            "backend": self.backend,
            "scores": self.scores,
        }


@dataclass
class ClassificationResult:
    """Document-level aggregated classification (Stage 2 output)."""

    label: str
    confidence: float
    backend: str
    page_predictions: list[PageClassification] = field(default_factory=list)
    aggregation: str = "majority_vote"
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "backend": self.backend,
            "aggregation": self.aggregation,
            "page_predictions": [p.to_dict() for p in self.page_predictions],
            "flags": list(self.flags),
        }


@dataclass
class ExtractionResult:
    """Stage 3 information extraction output."""

    fields: dict[str, list[str]] = field(default_factory=dict)
    fields_flat: dict[str, str | None] = field(default_factory=dict)
    backend: str = "none"
    document_type: str | None = None
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    page_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": self.fields,
            "fields_flat": self.fields_flat,
            "backend": self.backend,
            "document_type": self.document_type,
            "confidence": self.confidence,
            "flags": list(self.flags),
            "page_index": self.page_index,
        }


@dataclass
class DociePrediction:
    """
    Final aggregated output (Fig. 1 Output stage).

    Combines classification + extraction under application business rules
    and is the payload returned to callers / pushed downstream.
    """

    record_id: str
    application: str
    classification: ClassificationResult
    extraction: ExtractionResult
    processing: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    needs_human_review: bool = False
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "application": self.application,
            "classification": self.classification.to_dict(),
            "extraction": self.extraction.to_dict(),
            "processing": self.processing,
            "flags": list(dict.fromkeys(self.flags)),
            "needs_human_review": self.needs_human_review,
            "stage_timings_ms": self.stage_timings_ms,
            # Convenience mirrors for downstream consumers
            "document_type": self.classification.label,
            "fields": self.extraction.fields_flat,
        }

    def response_payload(self) -> dict[str, Any]:
        """Compact REST / downstream response shape (paper §VI)."""
        return {
            "record_id": self.record_id,
            "application": self.application,
            "document_type": self.classification.label,
            "classification_confidence": self.classification.confidence,
            "fields": self.extraction.fields_flat,
            "extraction_confidence": self.extraction.confidence,
            "needs_human_review": self.needs_human_review,
            "flags": list(dict.fromkeys(self.flags)),
        }
