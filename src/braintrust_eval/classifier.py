"""OpenRouter vision classifier used by Braintrust RVL-CDIP experiments.

Matches the historical ``CLASSIFICATION_PROMPT`` + Kimi K3 settings
(``max_tokens=500``, ``temperature=0.1``) and surfaces reasoning tokens when
OpenRouter returns them.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.rvl_cdip.openrouter_eval import label_list_str, normalize_rvl_label
from src.rvl_cdip.paths import LABEL_NAMES
from src.utils.config import Config
from src.utils.llm_client import OpenRouterClient
from src.utils.prompts import load_prompt

# Historical constant name from the external Braintrust PoC.
CLASSIFICATION_PROMPT = """\
You are classifying a scanned document page image into exactly one RVL-CDIP class.

Allowed labels (return exactly one of these strings, lowercase):
{label_list}

Rules:
- Look at layout, typography, and visual cues in the page image.
- Prefer distinctive structure (headers, tables, letterhead, columns) over OCR text alone.
- If two classes are plausible, pick the one that best matches the dominant page genre.
- Return ONLY the label text — no punctuation, no JSON, no explanation.

Label:
"""

SYSTEM_PROMPT = (
    "You are a careful document analysis assistant for insurance-style intake. "
    "Follow the task instructions exactly."
)

# Reasoning-capable OpenRouter slugs used by this experiment suite.
DEFAULT_VISION_REASONING_MODEL = "moonshotai/kimi-k3"
DEFAULT_PROMPT_IMPROVE_MODEL = "deepseek/deepseek-r1"
DEFAULT_FLAGSHIP_MODELS: tuple[str, ...] = (
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
)

DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.1


@dataclass
class ClassificationResult:
    document_id: str
    label: str
    prediction: str
    raw_text: str
    reasoning: str | None = None
    model_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_cached_tokens: int = 0
    completion_reasoning_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    error: str | None = None
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exact_match(self) -> bool:
        return (
            not self.error
            and normalize_rvl_label(self.prediction) == normalize_rvl_label(self.label)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_classification_prompt(
    *,
    prompt_template: str | None = None,
    labels: tuple[str, ...] = LABEL_NAMES,
) -> str:
    """Render the classification prompt with the official RVL label list."""
    template = prompt_template
    if template is None:
        try:
            template = load_prompt("rvl_classify_vision_bt")
        except FileNotFoundError:
            template = CLASSIFICATION_PROMPT
    return template.format(label_list=label_list_str(labels))


def default_reasoning_config(model_id: str) -> dict[str, Any] | None:
    """OpenRouter ``reasoning`` body for models that expose chain-of-thought.

    - Kimi K3: always thinks; request visible reasoning (``exclude=False``).
    - DeepSeek R1: enable reasoning at medium effort when used as text analyst.
    - Other models: ``None`` (no extra body).
    """
    slug = (model_id or "").strip().lower()
    if "kimi" in slug or "moonshot" in slug:
        return {"exclude": False, "effort": "high"}
    if "deepseek-r1" in slug or slug.endswith("/deepseek-r1"):
        return {"effort": "medium", "exclude": False}
    if any(x in slug for x in ("o1", "o3", "o4", "gpt-5")):
        return {"effort": "medium", "exclude": False}
    return None


def classify_image(
    *,
    image_path: Path,
    expected_label: str,
    document_id: str = "",
    model_id: str = DEFAULT_VISION_REASONING_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    prompt_template: str | None = None,
    cfg: Config | None = None,
    client: OpenRouterClient | None = None,
    dry_run: bool = False,
    reasoning: dict[str, Any] | None | object = ...,
) -> ClassificationResult:
    """Classify one fixed-size page image via OpenRouter.

    When ``dry_run=True``, returns the expected label without an API call
    (useful for wiring / CI). Pass ``reasoning=None`` to disable the OpenRouter
    reasoning body; omit it to use :func:`default_reasoning_config`.
    """
    doc_id = document_id or image_path.stem
    prompt = render_classification_prompt(prompt_template=prompt_template)

    if dry_run:
        return ClassificationResult(
            document_id=doc_id,
            label=expected_label,
            prediction=normalize_rvl_label(expected_label),
            raw_text=normalize_rvl_label(expected_label),
            reasoning="[dry-run] no model call",
            model_id=model_id,
            dry_run=True,
            metadata={"prompt_chars": len(prompt)},
        )

    config = cfg or Config.load()
    or_client = client or OpenRouterClient(model=model_id, cfg=config)
    if reasoning is ...:
        reasoning_body = default_reasoning_config(model_id)
    else:
        reasoning_body = reasoning  # type: ignore[assignment]

    start = time.perf_counter()
    try:
        resp = or_client.complete_multimodal(
            prompt,
            image=image_path,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=SYSTEM_PROMPT,
            preserve_square_png=True,
            reasoning=reasoning_body,
        )
        latency = time.perf_counter() - start
        usage = resp.get("usage") or {}
        raw = str(resp.get("text") or "")
        return ClassificationResult(
            document_id=doc_id,
            label=expected_label,
            prediction=normalize_rvl_label(raw),
            raw_text=raw,
            reasoning=resp.get("reasoning"),
            model_id=str(resp.get("model") or model_id),
            input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
            prompt_cached_tokens=int(usage.get("prompt_cached_tokens") or 0),
            completion_reasoning_tokens=int(usage.get("completion_reasoning_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            latency_seconds=float(latency),
            metadata={
                "finish_reason": resp.get("finish_reason"),
                "prompt_chars": len(prompt),
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    except Exception as exc:  # noqa: BLE001 — record per-row failure
        return ClassificationResult(
            document_id=doc_id,
            label=expected_label,
            prediction="",
            raw_text="",
            model_id=model_id,
            latency_seconds=time.perf_counter() - start,
            error=str(exc),
            metadata={"prompt_chars": len(prompt)},
        )
