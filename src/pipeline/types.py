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
    pdf_path: str | Path | None = None
    source_path: str | Path | None = None
    document_type_hint: str | None = None
    markdown: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "AnalysisDocument":
        record_id = str(row.get("record_id") or row.get("claim_id") or "unknown")
        text = str(row.get("text") or "")
        if not text and row.get("words"):
            text = " ".join(str(w.get("text") or "") for w in row["words"])
        source_path = row.get("source_path") or row.get("path") or row.get("file_path")
        image_path = row.get("image_path")
        pdf_path = row.get("pdf_path")
        # Infer from source_path when typed paths are absent
        if source_path and not image_path and not pdf_path:
            suffix = Path(str(source_path)).suffix.lower()
            if suffix == ".pdf":
                pdf_path = source_path
            elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
                image_path = source_path
        return cls(
            record_id=record_id,
            text=text,
            claim_id=row.get("claim_id"),
            image_path=image_path,
            pdf_path=pdf_path,
            source_path=source_path,
            document_type_hint=row.get("document_type"),
            markdown=row.get("markdown"),
            metadata={
                k: v
                for k, v in row.items()
                if k
                not in {
                    "record_id",
                    "text",
                    "claim_id",
                    "image_path",
                    "pdf_path",
                    "source_path",
                    "path",
                    "file_path",
                    "document_type",
                    "words",
                    "skeleton",
                    "markdown",
                }
            },
        )

    def llm_text(self) -> str:
        """Prefer compact markdown for LLM context; fall back to raw text."""
        if self.markdown and self.markdown.strip():
            return self.markdown
        return self.text or ""


@dataclass
class AnalysisContext:
    """
    Accumulating state passed through stages in initiation order.

    Each stage reads prior results and appends its own StageResult so later
    stages can react chronologically to upstream outputs.
    """

    document: AnalysisDocument
    stages: list[StageResult] = field(default_factory=list)
    markdown: dict[str, Any] | None = None
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
        # Only commit successful payloads into the slot dicts. Failed stages keep
        # prior slots as None/previous so downstream ``is None`` checks stay honest.
        if not result.ok:
            return
        if result.stage == "to_markdown":
            self.markdown = result.payload
            md = result.payload.get("markdown")
            if isinstance(md, str) and md.strip():
                self.document.markdown = md
                # Keep plain text in sync for encoder stages when source was image/pdf-only
                plain = result.payload.get("plain_text")
                if plain and not self.document.text.strip():
                    self.document.text = str(plain)
        elif result.stage == "classify":
            self.classification = result.payload
        elif result.stage == "extract":
            self.extraction = result.payload
        elif result.stage == "vision_llm":
            self.vision = result.payload
        elif result.stage == "summarize":
            self.summary = result.payload

    def content_for_llm(self) -> str:
        """Markdown-first content for generative / multimodal LLM prompts."""
        if self.markdown and self.markdown.get("markdown"):
            return str(self.markdown["markdown"])
        return self.document.llm_text()

    def content_for_encoder(self) -> str:
        """Plain-ish text for classification / token extraction encoders."""
        if self.markdown and self.markdown.get("plain_text"):
            return str(self.markdown["plain_text"])
        if self.document.markdown:
            from src.pipeline.markdown_convert import markdown_to_plain

            return markdown_to_plain(self.document.markdown)
        return self.document.text or ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.document.record_id,
            "claim_id": self.document.claim_id,
            "markdown": self.markdown,
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
                s.confidence < 0.55 and s.ok
                for s in self.stages
                if s.stage in {"classify", "extract", "to_markdown"}
            ),
        }
