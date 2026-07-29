"""Use DeepSeek R1 (or another reasoning text model) to improve the classify prompt.

Takes a local predictions JSONL (from ``run_local_loop`` / Braintrust export),
selects misclassifications that include reasoning traces, and asks the model to
propose an improved ``CLASSIFICATION_PROMPT``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.braintrust_eval.classifier import (
    CLASSIFICATION_PROMPT,
    DEFAULT_PROMPT_IMPROVE_MODEL,
    UNDERSCORE_LABELS,
    default_reasoning_config,
    normalize_capstone_label,
    render_classification_prompt,
)
from src.utils.config import Config
from src.utils.llm_client import OpenRouterClient
from src.utils.prompts import load_prompt

IMPROVE_SYSTEM = (
    "You are an expert at writing vision-LLM classification prompts for "
    "scanned business documents (RVL-CDIP taxonomy). Return an improved prompt "
    "that lists the 16 underscore class names with short descriptions and still "
    "requires the model to output ONLY the lowercase underscore class name."
)

IMPROVE_USER = """\
Current classification prompt template:
---
{current_prompt}
---

Official underscore labels: {labels}

Misclassification examples (gold → predicted) with model reasoning when available:
{error_block}

Write an improved prompt that:
1. Keeps all 16 underscore class names (file_folder, news_article, etc.).
2. Adds brief, high-signal visual discriminators for commonly confused pairs.
3. Still ends by asking for ONLY the class name (no JSON / explanation).
4. Stays concise (prefer under 500 words).

Return ONLY the new prompt text.
"""


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_errors(
    predictions: Sequence[dict[str, Any]],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    errors = [
        r
        for r in predictions
        if not r.get("error")
        and normalize_capstone_label(r.get("prediction"))
        != normalize_capstone_label(r.get("label"))
    ]
    # Prefer rows that have reasoning text for analysis.
    errors.sort(key=lambda r: (0 if r.get("reasoning") else 1, r.get("document_id") or ""))
    return errors[:limit]


def format_error_block(errors: Sequence[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for i, row in enumerate(errors, start=1):
        reasoning = (row.get("reasoning") or "").strip()
        if len(reasoning) > 1200:
            reasoning = reasoning[:1200] + "…"
        chunks.append(
            f"{i}. gold={row.get('label')!r} pred={row.get('prediction')!r} "
            f"doc={row.get('document_id')}\n"
            f"   reasoning: {reasoning or '[none captured]'}"
        )
    return "\n".join(chunks) if chunks else "(no errors supplied)"


def improve_prompt(
    *,
    predictions_path: Path,
    model_id: str = DEFAULT_PROMPT_IMPROVE_MODEL,
    limit: int = 24,
    cfg: Config | None = None,
    dry_run: bool = False,
    current_prompt: str | None = None,
) -> dict[str, Any]:
    """Ask a reasoning text model to revise the classification prompt."""
    preds = _load_predictions(predictions_path)
    errors = select_errors(preds, limit=limit)
    try:
        base_prompt = current_prompt or load_prompt("rvl_classify_vision_bt")
    except FileNotFoundError:
        base_prompt = current_prompt or CLASSIFICATION_PROMPT

    user = IMPROVE_USER.format(
        current_prompt=base_prompt,
        labels=", ".join(UNDERSCORE_LABELS),
        error_block=format_error_block(errors),
    )

    if dry_run:
        return {
            "model_id": model_id,
            "n_errors_used": len(errors),
            "improved_prompt": base_prompt,
            "dry_run": True,
            "note": "dry_run — returned current prompt unchanged",
        }

    config = cfg or Config.load()
    client = OpenRouterClient(model=model_id, cfg=config)
    resp = client.complete(
        user,
        max_tokens=2000,
        temperature=0.2,
        system_prompt=IMPROVE_SYSTEM,
        reasoning=default_reasoning_config(model_id),
    )
    improved = str(resp.get("text") or "").strip()

    return {
        "model_id": str(resp.get("model") or model_id),
        "n_errors_used": len(errors),
        "improved_prompt": improved,
        "reasoning": resp.get("reasoning"),
        "usage": resp.get("usage"),
        "dry_run": False,
        "rendered_preview": render_classification_prompt(prompt_template=improved)[:500],
    }
