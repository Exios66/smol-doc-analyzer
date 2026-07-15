"""
Local fine-tuned model entrypoints for the eval harness.

Wires classification / extraction / memo_generation tasks to the same pipeline
stages used in production inference (DeBERTa classifier, LayoutLM extractor,
template or LoRA summarizer). Swap individual callables if you serve models
via vLLM instead of in-process transformers.
"""

from __future__ import annotations

from typing import Any, Callable

from src.pipeline.stages import ClassifyStage, ExtractStage, MarkdownConvertStage, SummarizeStage
from src.pipeline.types import AnalysisContext, AnalysisDocument
from src.utils.config import Config


def _document_from_example(example: dict[str, Any]) -> AnalysisDocument:
    fields = example.get("prompt_fields") or {}
    text = str(
        fields.get("document_text")
        or fields.get("text")
        or example.get("text")
        or ""
    )
    markdown = fields.get("markdown") or example.get("markdown")
    return AnalysisDocument(
        record_id=str(example.get("example_id") or example.get("record_id") or "eval"),
        text=text,
        claim_id=example.get("claim_id") or fields.get("claim_id"),
        image_path=fields.get("image_path") or example.get("image_path"),
        pdf_path=fields.get("pdf_path") or example.get("pdf_path"),
        markdown=markdown if isinstance(markdown, str) else None,
        document_type_hint=fields.get("document_type_hint"),
        metadata={"eval_task": example.get("task")},
    )


def _ctx_with_markdown(cfg: Config, example: dict[str, Any]) -> AnalysisContext:
    doc = _document_from_example(example)
    ctx = AnalysisContext(document=doc)
    # Prefer pre-baked markdown from the eval set; otherwise run the convert stage.
    if doc.markdown:
        ctx.markdown = {
            "markdown": doc.markdown,
            "plain_text": doc.text or doc.markdown,
            "backend": "eval_set",
            "approx_tokens": max(1, len(doc.markdown) // 4),
        }
        return ctx
    md_stage = MarkdownConvertStage(cfg=cfg)
    ctx.add(md_stage.run(ctx))
    return ctx


def build_local_task_endpoints(cfg: Config | None = None) -> dict[str, Callable[[dict], Any]]:
    """Return task -> callable(example) -> prediction mappings."""
    cfg = cfg or Config.load()
    classify = ClassifyStage(cfg=cfg)
    extract = ExtractStage(cfg=cfg, render_image=False)
    summarize = SummarizeStage(cfg=cfg)

    # Warm weights once so the first billed example does not absorb cold-start load.
    try:
        classify._ensure_loaded()
    except Exception:
        pass
    try:
        extract._ensure_loaded()
    except Exception:
        pass
    try:
        summarize._ensure_loaded()
    except Exception:
        pass

    def classification(example: dict[str, Any]) -> str:
        ctx = _ctx_with_markdown(cfg, example)
        result = classify.run(ctx)
        if not result.ok:
            raise RuntimeError(result.error or "local classification failed")
        return str(result.payload.get("document_type") or "").strip().lower()

    def extraction(example: dict[str, Any]) -> dict[str, Any]:
        ctx = _ctx_with_markdown(cfg, example)
        # Extraction can react to a classification hint when present.
        hint = (example.get("prompt_fields") or {}).get("document_type_hint")
        if hint:
            ctx.classification = {"document_type": hint, "backend": "eval_hint", "confidence": 1.0}
        else:
            clf = classify.run(ctx)
            ctx.add(clf)
            if not clf.ok:
                raise RuntimeError(clf.error or "local classification failed before extraction")
        result = extract.run(ctx)
        if not result.ok:
            raise RuntimeError(result.error or "local extraction failed")
        # Prefer flat field map for apples-to-apples comparison with frontier JSON.
        flat = result.payload.get("fields_flat")
        if isinstance(flat, dict) and flat:
            return flat
        fields = result.payload.get("fields") or {}
        return fields if isinstance(fields, dict) else {"_raw": fields}

    def memo_generation(example: dict[str, Any]) -> str:
        ctx = _ctx_with_markdown(cfg, example)
        fields = example.get("prompt_fields") or {}
        # Allow eval examples to supply upstream payloads so memo gen can run
        # without re-extracting when the eval set already has gold fields.
        if fields.get("document_type"):
            ctx.classification = {
                "document_type": fields["document_type"],
                "backend": "eval_hint",
                "confidence": 1.0,
            }
        else:
            clf = classify.run(ctx)
            ctx.add(clf)
            if not clf.ok:
                raise RuntimeError(clf.error or "local classification failed before memo generation")

        if isinstance(fields.get("extracted_fields"), dict):
            flat = fields["extracted_fields"]
            ctx.extraction = {
                "fields_flat": flat,
                "fields": {k: [v] for k, v in flat.items()},
                "backend": "eval_hint",
            }
        else:
            ext = extract.run(ctx)
            ctx.add(ext)
            if not ext.ok:
                raise RuntimeError(ext.error or "local extraction failed before memo generation")

        result = summarize.run(ctx)
        if not result.ok:
            raise RuntimeError(result.error or "local memo generation failed")
        return str(result.payload.get("memo") or "").strip()

    return {
        "classification": classification,
        "extraction": extraction,
        "memo_generation": memo_generation,
    }
