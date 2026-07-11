"""Format pipeline analysis results for Discord replies."""

from __future__ import annotations

from typing import Any


def compact_analysis(result: dict[str, Any], *, max_memo_chars: int = 1200) -> dict[str, Any]:
    """Reduce a full AnalysisContext.to_dict() payload for the agent / Discord."""
    classification = result.get("classification") or {}
    extraction = result.get("extraction") or {}
    vision = result.get("vision") or {}
    outcome = result.get("outcome") or {}
    summary = result.get("summary") or {}
    markdown = result.get("markdown") or {}

    fields = extraction.get("fields") or extraction.get("extracted_fields") or {}
    refined = vision.get("refined_fields") or {}
    if refined:
        # Prefer refined fields when present.
        merged = dict(fields)
        merged.update({k: v for k, v in refined.items() if v not in (None, "", [])})
        fields = merged

    memo = result.get("memo") or summary.get("memo") or ""
    if isinstance(memo, str) and len(memo) > max_memo_chars:
        memo = memo[: max_memo_chars - 1].rstrip() + "…"

    stage_summaries = []
    for stage in result.get("stages") or []:
        stage_summaries.append(
            {
                "stage": stage.get("stage"),
                "ok": stage.get("ok"),
                "confidence": round(float(stage.get("confidence") or 0.0), 3),
                "flags": stage.get("flags") or [],
                "error": stage.get("error"),
            }
        )

    return {
        "record_id": result.get("record_id"),
        "claim_id": result.get("claim_id"),
        "document_type": classification.get("document_type")
        or classification.get("label")
        or classification.get("predicted_label"),
        "classification_confidence": classification.get("confidence"),
        "expected_outcome": result.get("expected_outcome")
        or outcome.get("expected_outcome")
        or outcome.get("outcome_label"),
        "outcome_confidence": outcome.get("confidence"),
        "outcome_description": outcome.get("description"),
        "gold_outcome": outcome.get("gold_outcome"),
        "outcome_correct": outcome.get("correct"),
        "fields": fields,
        "memo": memo,
        "flags": result.get("flags") or [],
        "low_confidence": bool(result.get("low_confidence")),
        "markdown_tokens": markdown.get("approx_tokens"),
        "stages": stage_summaries,
    }


def format_discord_summary(compact: dict[str, Any]) -> str:
    """Human-readable Discord markdown from a compact analysis dict."""
    lines: list[str] = ["## Document analysis"]

    doc_type = compact.get("document_type") or "unknown"
    conf = compact.get("classification_confidence")
    conf_s = f" ({float(conf):.0%})" if isinstance(conf, (int, float)) else ""
    lines.append(f"**Type:** `{doc_type}`{conf_s}")

    if compact.get("claim_id") or compact.get("record_id"):
        rid = compact.get("claim_id") or compact.get("record_id")
        lines.append(f"**Record:** `{rid}`")

    outcome = compact.get("expected_outcome")
    if outcome:
        oconf = compact.get("outcome_confidence")
        oconf_s = f" ({float(oconf):.0%})" if isinstance(oconf, (int, float)) else ""
        lines.append(f"**Predicted outcome:** `{outcome}`{oconf_s}")
        if compact.get("outcome_description"):
            lines.append(f"-# {compact['outcome_description']}")
        if compact.get("gold_outcome") is not None:
            mark = "✓" if compact.get("outcome_correct") else "✗"
            lines.append(f"**Gold outcome:** `{compact['gold_outcome']}` {mark}")

    fields = compact.get("fields") or {}
    if fields:
        lines.append("")
        lines.append("### Extracted fields")
        for key, value in list(fields.items())[:24]:
            lines.append(f"- **{key}:** {value}")
        if len(fields) > 24:
            lines.append(f"- …and {len(fields) - 24} more")

    memo = compact.get("memo")
    if memo:
        lines.append("")
        lines.append("### Memo")
        lines.append(str(memo))

    flags = compact.get("flags") or []
    if flags or compact.get("low_confidence"):
        lines.append("")
        lines.append("### Review flags")
        if compact.get("low_confidence"):
            lines.append("- low_confidence")
        for flag in flags[:12]:
            lines.append(f"- {flag}")

    return "\n".join(lines)
