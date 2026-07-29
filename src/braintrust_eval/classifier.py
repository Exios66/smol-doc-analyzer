"""OpenRouter vision classifier used by Braintrust RVL-CDIP experiments.

Ports ``CLASSIFICATION_PROMPT`` from the AMFAM Doc Intel capstone
(https://github.com/grantmooslin/AMFAM_Doc_intel_capstone) with underscore
class names, plus Kimi K3 settings (``max_tokens=500``, ``temperature=0.1``)
and OpenRouter reasoning-token capture.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.rvl_cdip.paths import LABEL_NAMES
from src.utils.config import Config
from src.utils.llm_client import OpenRouterClient
from src.utils.prompts import load_prompt

# Capstone / Braintrust class ids (underscore form). Order matches RVL label_ids.
UNDERSCORE_LABELS: tuple[str, ...] = tuple(
    name.replace(" ", "_") for name in LABEL_NAMES
)

VALID_CLASSES: tuple[str, ...] = (
    "advertisement",
    "budget",
    "email",
    "file_folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news_article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific_publication",
    "scientific_report",
    "specification",
)

# Strengthened RVL-CDIP prompt: purpose-first boundaries + tie-breakers for
# budget/presentation/specification/scientific_report/questionnaire confusions.
# Keep in sync with evaluation/prompts/rvl_classify_vision_bt.txt.
CLASSIFICATION_PROMPT = """\
You are a document classification expert analyzing document images with a vision model. Classify the given image into one of these 16 categories:

Valid labels (return exactly one, lowercase underscore form):
letter, form, email, handwritten, advertisement, scientific_report, scientific_publication, specification, file_folder, news_article, budget, invoice, presentation, questionnaire, resume, memo

Available Classes:
advertisement - Marketing materials, promotional content, flyers, brochures, catalog/sales promotion (even if letter-formatted)
budget - Financial planning, estimates, retainers, contribution requests, expense summaries, payment allocation, projected amounts, price/value estimates, handwritten financial tallies
email - Email messages, email threads, electronic correspondence
file_folder - File folder labels, directory listings, file organization documents
form - Structured data-entry or application forms that are not better classified as questionnaire, budget, invoice, specification, or resume
handwritten - Handwritten notes, manuscripts, or handwritten correspondence (primary identity is handwriting)
invoice - Bills, invoices, receipts, remittance/payment requests, balance-due documents clearly requesting payment for goods/services
letter - Ordinary correspondence where the primary purpose is communication rather than promotion
memo - Memorandums, internal communications, office memos
news_article - Newspaper articles, news reports, journalistic content
presentation - Title slides, training decks, program/agenda/strategy slides, sparse large-heading presentation pages
questionnaire - Surveys, screeners, interview scripts, rating scales, skip logic, respondent fields, documents collecting answers
resume - CVs, resumes, job applications, professional profiles
scientific_publication - Published journal/conference-style academic articles with title, authors, abstract, journal/proceedings metadata, citations, or publication formatting
scientific_report - Internal reports, lab reports, analytical data summaries, grant sections, preliminary studies, technical reports, or non-published scientific documentation
specification - Product specs, MSDS/safety data sheets, technical parameter sheets, manufacturing requirements, formula/component sheets, product change/spec documents

Input Data:
- Document image (300 DPI grayscale)

Classification policy:
Classify by the document's primary purpose, not just its visual layout or medium.

Important boundary rules:
- budget: Prefer budget over form/invoice/handwritten when the main purpose is financial planning, estimates, retainers, contribution requests, expense summaries, or handwritten financial tallies. Do not choose invoice unless the document clearly requests payment for goods/services or has an invoice/bill/remittance/balance-due structure.
- presentation: Prefer presentation over memo/form/news_article/letter if the page appears to be a title slide, training deck page, agenda/program slide, or sparse large-title business presentation page — even if the text looks memo-like or news-like.
- specification: Prefer specification over form/scientific_report for MSDS/safety data sheets, product sheets, technical parameter sheets, manufacturing requirements, and product change/spec documents. Do not choose form just because fields are structured, or scientific_report just because content is technical/chemical.
- scientific_publication vs scientific_report: Choose scientific_publication only for published academic/journal/conference-style papers. Choose scientific_report for internal/lab/grant/analytical/non-published scientific documentation.
- questionnaire: Prefer questionnaire over form/letter/handwritten when the page collects responses via questions, rating scales, checkboxes, skip logic, or survey/interview fields — even if handwritten or accompanied by a cover letter.
- handwritten: Choose handwritten only when the document's primary identity is a handwritten note, manuscript, or handwritten correspondence — not when handwriting is merely responses on a questionnaire/form or financial notes better labeled budget.
- advertisement vs letter: If formatted as a letter but the main purpose is marketing/promotion/product sales/catalog advertising, choose advertisement.

Tie-breaker rule:
When multiple labels seem plausible, choose the label that best describes the document's purpose in this dataset, not merely its layout.
Examples:
- A financial estimate on a structured form is budget, not form.
- A survey with handwritten answers is questionnaire, not handwritten.
- A technical safety sheet is specification, not form.
- A title page for training/compliance material is presentation, not memo.

Common confusion priorities:
- Financial planning/estimate/retainer/contribution/expense data → prefer budget over form, invoice, or handwritten.
- Slide/title/training/program page → prefer presentation over memo, letter, form, or news_article.
- Technical product/manufacturing/safety data → prefer specification over form or scientific_report.
- Survey/interview/question-response document → prefer questionnaire over form, letter, or handwritten.
- Published journal/conference article → prefer scientific_publication.
- Internal/lab/grant/technical scientific documentation → prefer scientific_report.

Few-shot examples (weak / ambiguous classes):
Example: Document "Outdoor Estimate Recap" with client, estimate number, totals, product/month amounts. → budget
Example: Handwritten list of brands with dollar amounts and a total. → budget
Example: Political campaign contribution request with amount, account code, and approval fields. → budget
Example: "Master Settlement Agreement Compliance Training" title page. → presentation
Example: "Retail Leaders Performance Measurement / Market Walk 2001" large sparse title page. → presentation
Example: Internal business page with program title and deck-like layout. → presentation
Example: Material Safety Data Sheet with chemical hazards, physical data, and manufacturer details. → specification
Example: Cigarette/filter rod specification with length, circumference, paper, adhesive, and technical parameters. → specification
Example: Product change document listing technical components and approvals. → specification
Example: Published journal article with title, authors, abstract, journal name, and citations. → scientific_publication
Example: Internal preliminary studies section from a grant or lab report. → scientific_report
Example: Analytical data summary with HPLC, UV/VIS, NMR, and mass spectrum results. → scientific_report
Example: Survey/screener with rating scales or checkboxes, possibly with handwritten answers. → questionnaire

Analysis Approach (decide internally, do not write these steps):
1. What is the document's primary purpose?
2. Is it financial, presentation, specification, survey, scientific, correspondence, or advertising?
3. Apply the tie-breaker / confusion-priority rules for ambiguous cases.
4. Return only the final class label.

Output:
Output only the class name. No explanation, no JSON, no additional text. Do not include reasoning, punctuation, markdown, or extra text.

Example: If the document has "INVOICE" header, line items table, and total amount, output only:
invoice"""

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


def to_underscore_label(value: Any) -> str:
    """Canonicalize spaced or underscore RVL labels to capstone underscore form."""
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in VALID_CLASSES or raw in UNDERSCORE_LABELS:
        return raw.replace(" ", "_")
    spaced = raw.replace("_", " ")
    if spaced in LABEL_NAMES:
        return spaced.replace(" ", "_")
    # Tolerate "scientific report" / mixed punctuation
    compact = " ".join(raw.replace("_", " ").split())
    if compact in LABEL_NAMES:
        return compact.replace(" ", "_")
    return raw.replace(" ", "_")


def clean_prediction(text: str | None) -> str:
    """Extract a valid underscore class name from LLM response (capstone logic)."""
    if not text:
        return ""
    lowered = text.strip().lower().replace("-", "_")
    # Prefer longest class match so scientific_publication beats publication, etc.
    for cls in sorted(VALID_CLASSES, key=len, reverse=True):
        if cls in lowered or cls.replace("_", " ") in lowered:
            return cls
    return to_underscore_label(lowered.splitlines()[0] if lowered else "")


def normalize_capstone_label(value: Any) -> str:
    """Map gold or predicted labels onto underscore class names for scoring."""
    cleaned = clean_prediction(str(value or ""))
    if cleaned in VALID_CLASSES:
        return cleaned
    return to_underscore_label(value)


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
            and normalize_capstone_label(self.prediction)
            == normalize_capstone_label(self.label)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_classification_prompt(*, prompt_template: str | None = None) -> str:
    """Load the capstone classification prompt (no format placeholders)."""
    if prompt_template is not None:
        # Allow improved prompts that still use {label_list}.
        if "{label_list}" in prompt_template:
            label_list = "\n".join(f"- {name}" for name in UNDERSCORE_LABELS)
            return prompt_template.format(label_list=label_list)
        return prompt_template
    try:
        return load_prompt("rvl_classify_vision_bt")
    except FileNotFoundError:
        return CLASSIFICATION_PROMPT


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
    gold = normalize_capstone_label(expected_label)

    if dry_run:
        return ClassificationResult(
            document_id=doc_id,
            label=gold,
            prediction=gold,
            raw_text=gold,
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
            label=gold,
            prediction=clean_prediction(raw),
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
            label=gold,
            prediction="",
            raw_text="",
            model_id=model_id,
            latency_seconds=time.perf_counter() - start,
            error=str(exc),
            metadata={"prompt_chars": len(prompt)},
        )
