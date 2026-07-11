"""Shared types for the chained document-analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    """Output produced by one pipeline stage."""

    stage: str
    order: int
    ok: bool
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class AnalysisDocument:
    """Inbound document for a single analysis action."""

    record_id: str
    text: str
    claim_id: str | None = None
    image_path: str | Path | None = None
    document_type_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "AnalysisDocument":
        record_id = str(row.get("record_id") or row.get("claim_id") or "unknown")
        text = str(row.get("text") or "")
        if not text and row.get("words"):
            text = " ".join(w.get("text", "") for w in row["words"])
        return cls(
            record_id=record_id,
            text=text,
            claim_id=row.get("claim_id"),
            image_path=row.get("image_path"),
            document_type_hint=row.get("document_type"),
            metadata={
                k: v
                for k, v in row.items()
                if k
                not in {
                    "record_id",
                    "text",
                    "claim_id",
                    "image_path",
                    "document_type",
                    "words",
                    "skeleton",
                }
            },
        )


@dataclass
class AnalysisContext:
    """
    Accumulating state passed through stages in initiation order.

    Each stage reads prior results and appends its own StageResult so later
    stages can react chronologically to upstream outputs.
    """

    document: AnalysisDocument
    stages: list[StageResult] = field(default_factory=list)
    classification: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)

    def prior(self, stage_name: str) -> StageResult | None:
        for result in self.stages:
            if result.stage == stage_name:
                return result
        return None

    def add(self, result: StageResult) -> None:
        self.stages.append(result)
        self.flags.extend(result.flags)
        if result.stage == "classify":
            self.classification = result.payload
        elif result.stage == "extract":
            self.extraction = result.payload
        elif result.stage == "vision_llm":
            self.vision = result.payload
        elif result.stage == "summarize":
            self.summary = result.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.document.record_id,
            "claim_id": self.document.claim_id,
            "classification": self.classification,
            "extraction": self.extraction,
            "vision": self.vision,
            "summary": self.summary,
            "flags": list(dict.fromkeys(self.flags)),
            "stages": [
                {
                    "stage": s.stage,
                    "order": s.order,
                    "ok": s.ok,
                    "confidence": s.confidence,
                    "flags": s.flags,
                    "error": s.error,
                    "payload": s.payload,
                }
                for s in self.stages
            ],
            "memo": (self.summary or {}).get("memo"),
            "low_confidence": any(
                s.confidence < 0.55 and s.ok for s in self.stages if s.stage in {"classify", "extract"}
            ),
        }
