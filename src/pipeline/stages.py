"""Pipeline stage protocol and concrete analysis stages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from src.extraction.render_forms import FIELD_PATTERNS, label_words, render_page
from src.pipeline.markdown_convert import approx_token_count, convert_to_markdown
from src.pipeline.outcome import OUTCOME_LABELS, predict_outcome
from src.pipeline.types import AnalysisContext, StageResult
from src.utils.config import Config
from src.utils.io import read_json

logger = logging.getLogger(__name__)

# Confidence below this surfaces a human-review flag rather than failing the chain.
LOW_CONFIDENCE = 0.55


class PipelineStage(Protocol):
    name: str
    order: int

    def run(self, ctx: AnalysisContext) -> StageResult: ...


def _taxonomy_labels(cfg: Config) -> list[str]:
    tax = yaml.safe_load(cfg.taxonomy_path.read_text(encoding="utf-8"))
    return [c["label"] for c in tax["categories"]]


def _heuristic_classify(text: str, labels: list[str]) -> tuple[str, float]:
    """Keyword fallback when no classifier weights are available."""
    lower = text.lower()
    scores: dict[str, float] = {lab: 0.0 for lab in labels}
    rules = [
        ("application_commercial", ["commercial insurance application", "acord form: 125", "coverage sections"]),
        ("application_personal", ["personal lines application", "acord form: 90", "vehicle / property"]),
        ("certificate_evidence", ["certificate of insurance", "evidence of insurance", "acord form: 25"]),
        ("loss_notice", ["loss notice", "date of loss", "acord form: 1", "acord form: 2"]),
        ("claims_correspondence", ["dear", "regarding your claim", "status update"]),
        ("adjuster_memo", ["adjuster memo", "next steps", "to: claims file"]),
        ("policy_change_endorsement", ["endorsement", "policy change", "acord form: 101"]),
        ("repair_estimate", ["repair estimate", "labor", "parts", "estimate total"]),
        ("supporting_evidence", ["police report", "inspection report", "photo description"]),
    ]
    for label, keys in rules:
        if label not in scores:
            continue
        hits = sum(1 for k in keys if k in lower)
        scores[label] = hits / max(len(keys), 1)
    best = max(scores, key=scores.get)
    conf = float(scores[best])
    if conf <= 0:
        # Prefer loss_notice when loss fields dominate, else first taxonomy label.
        if "date of loss" in lower or "loss type" in lower:
            return "loss_notice", 0.35
        return labels[0], 0.2
    return best, min(0.95, 0.4 + conf * 0.6)


@dataclass
class MarkdownConvertStage:
    """
    First stage: PNG / PDF / text → compact structured markdown.

    Downstream LLM stages consume this markdown instead of raw page images,
    cutting vision/token cost while preserving headings and field structure.
    """

    cfg: Config
    order: int = 0
    name: str = "to_markdown"

    def run(self, ctx: AnalysisContext) -> StageResult:
        flags: list[str] = []
        try:
            # Persist converted markdown under pipeline cache for inspection / reuse
            cache_dir = self.cfg.pipeline_cache_dir / "markdown"
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_id = ctx.document.record_id.replace("::", "__").replace("/", "_")

            conversion = convert_to_markdown(
                text=ctx.document.text or None,
                image_path=ctx.document.image_path,
                pdf_path=ctx.document.pdf_path,
                source_path=ctx.document.source_path,
            )
            md_path = cache_dir / f"{safe_id}.md"
            md_path.write_text(conversion.markdown, encoding="utf-8")

            raw_chars = len(ctx.document.text or "")
            raw_tokens = approx_token_count(ctx.document.text or "")
            # Rough image-token stand-in: a page image often costs ~1k–2k+ vision tokens
            image_token_proxy = 1600 if (ctx.document.image_path or ctx.document.pdf_path) else 0
            baseline_tokens = max(raw_tokens, image_token_proxy)
            saved = max(0, baseline_tokens - conversion.approx_tokens)

            if conversion.source_kind in {"png", "pdf", "image"}:
                flags.append(f"markdown_from_{conversion.source_kind}")
            if conversion.backend in {"failed", "empty", "none", "pytesseract_failed"}:
                flags.append("low_confidence_markdown")
            if not conversion.plain_text.strip() and not conversion.markdown.strip():
                flags.append("markdown_empty")

            confidence = 0.9
            if conversion.source_kind == "empty":
                confidence = 0.1
            elif conversion.backend.endswith("failed") or conversion.backend == "none":
                confidence = 0.35
            elif conversion.extras and conversion.extras.get("used_fallback_text"):
                confidence = 0.7
                flags.append("markdown_image_fallback_text")

            payload = conversion.to_dict()
            payload["markdown_path"] = str(md_path)
            payload["token_baseline"] = baseline_tokens
            payload["token_saved_est"] = saved
            payload["fed_to_llm_as"] = "markdown"

            return StageResult(
                stage=self.name,
                order=self.order,
                ok=True,
                confidence=confidence,
                flags=flags,
                payload=payload,
            )
        except Exception as exc:
            logger.exception("Markdown conversion failed")
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=False,
                confidence=0.0,
                flags=["markdown_failed"],
                error=str(exc),
                payload={},
            )


@dataclass
class ClassifyStage:
    """Document-type classification (DeBERTa / DistilBERT, with heuristic fallback)."""

    cfg: Config
    model_dir: Path | None = None
    order: int = 1
    name: str = "classify"
    _model: Any = None
    _tokenizer: Any = None
    _id2label: dict[int, str] | None = None
    _backend: str = "unloaded"

    def _ensure_loaded(self) -> None:
        if self._backend != "unloaded":
            return
        path = self.model_dir or (self.cfg.models_dir / "classifier_smoke")
        if not path.exists() or not (path / "config.json").exists():
            # Prefer non-smoke classifier if present
            alt = self.cfg.models_dir / "classifier"
            path = alt if (alt / "config.json").exists() else path
        if not path.exists() or not (path / "config.json").exists():
            self._backend = "heuristic"
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(path))
            self._model.eval()
            label2id = read_json(path / "label2id.json")
            self._id2label = {int(v): k for k, v in label2id.items()}
            self._torch = torch
            self._backend = "transformers"
            self.model_dir = path
        except Exception as exc:
            logger.warning("Classifier load failed (%s); using heuristic", exc)
            self._backend = "heuristic"

    def run(self, ctx: AnalysisContext) -> StageResult:
        self._ensure_loaded()
        labels = _taxonomy_labels(self.cfg)
        flags: list[str] = []
        text = ctx.content_for_encoder()
        try:
            if self._backend == "transformers":
                assert self._model is not None and self._tokenizer is not None
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with self._torch.no_grad():
                    logits = self._model(**inputs).logits[0]
                    probs = self._torch.softmax(logits, dim=-1)
                    pred_id = int(self._torch.argmax(probs).item())
                    confidence = float(probs[pred_id].item())
                label = self._id2label.get(pred_id, labels[0])
                backend = "transformers"
            else:
                label, confidence = _heuristic_classify(text, labels)
                backend = "heuristic"
                flags.append("classify_heuristic")

            if confidence < LOW_CONFIDENCE:
                flags.append("low_confidence_classification")

            return StageResult(
                stage=self.name,
                order=self.order,
                ok=True,
                confidence=confidence,
                flags=flags,
                payload={
                    "document_type": label,
                    "confidence": confidence,
                    "backend": backend,
                    "input_from": "markdown" if ctx.markdown else "text",
                    "model_dir": str(self.model_dir) if self.model_dir else None,
                },
            )
        except Exception as exc:
            logger.exception("Classify stage failed")
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=False,
                confidence=0.0,
                flags=["classify_failed"],
                error=str(exc),
                payload={},
            )


def _heuristic_extract(text: str) -> dict[str, list[str]]:
    """Regex / prefix extraction aligned with FIELD_PATTERNS (+ markdown tables)."""
    fields: dict[str, list[str]] = {}
    for field, prefix in FIELD_PATTERNS:
        # Match "Prefix: value" on a line
        pat = re.compile(re.escape(prefix) + r"\s*(.+)", re.IGNORECASE)
        for line in text.splitlines():
            m = pat.search(line.strip())
            if m:
                value = m.group(1).strip()
                if value:
                    fields.setdefault(field, []).append(value)

    # Markdown table rows: | **Date Of Loss** | 2023-02-17 |
    table_pat = re.compile(
        r"^\|\s*\*?\*?([^*\n|]+?)\*?\*?\s*\|\s*(.*?)\s*\|$",
        re.MULTILINE,
    )
    label_to_field = {
        field.replace("_", " ").lower(): field for field, _ in FIELD_PATTERNS
    }
    for prefix, field in ((p, f) for f, p in FIELD_PATTERNS):
        label_to_field[prefix.rstrip(":").lower()] = field
        label_to_field[field.lower()] = field

    for match in table_pat.finditer(text):
        label = match.group(1).strip().lower()
        value = match.group(2).strip()
        if label in {"field", "---"} or not value or value == "---":
            continue
        field = label_to_field.get(label)
        if field and value:
            fields.setdefault(field, []).append(value)
    return fields


@dataclass
class ExtractStage:
    """
    Field extraction — reacts to classification output.

    Uses a trained token classifier when available; otherwise heuristic
    FIELD_PATTERNS extraction. Optionally renders text to an image for
    layout-aware / Vision LLM downstream stages.
    """

    cfg: Config
    model_dir: Path | None = None
    order: int = 2
    name: str = "extract"
    render_image: bool = True
    _model: Any = None
    _tokenizer: Any = None
    _id2label: dict[int, str] | None = None
    _backend: str = "unloaded"

    def _ensure_loaded(self) -> None:
        if self._backend != "unloaded":
            return
        path = self.model_dir or (self.cfg.models_dir / "extractor_smoke")
        if not (path / "config.json").exists():
            alt = self.cfg.models_dir / "extractor"
            path = alt if (alt / "config.json").exists() else path
        if not (path / "config.json").exists():
            self._backend = "heuristic"
            return
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(path))
            self._model = AutoModelForTokenClassification.from_pretrained(str(path))
            self._model.eval()
            label2id = {k: int(v) for k, v in read_json(path / "label2id.json").items()}
            self._id2label = {v: k for k, v in label2id.items()}
            self._torch = torch
            self._backend = "transformers"
            self.model_dir = path
        except Exception as exc:
            logger.warning("Extractor load failed (%s); using heuristic", exc)
            self._backend = "heuristic"

    def _model_extract(self, tokens: list[str]) -> dict[str, list[str]]:
        from src.extraction.eval import _decode_entities

        assert self._model is not None and self._tokenizer is not None and self._id2label is not None
        enc = self._tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with self._torch.no_grad():
            logits = self._model(**enc).logits[0]
            pred_ids = self._torch.argmax(logits, dim=-1).tolist()
        word_ids = enc.word_ids(batch_index=0)
        aligned = ["O"] * len(tokens)
        seen: set[int] = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            aligned[wid] = self._id2label.get(int(pred_ids[idx]), "O")
        return dict(_decode_entities(tokens, aligned))

    def run(self, ctx: AnalysisContext) -> StageResult:
        self._ensure_loaded()
        flags: list[str] = []
        doc_type = (ctx.classification or {}).get("document_type") or ctx.document.document_type_hint
        source_text = ctx.content_for_encoder() or ctx.document.text
        # Prefer markdown for heuristic field tables when available
        md_text = ctx.content_for_llm() if ctx.markdown else source_text
        try:
            # Ensure an image exists for layout / vision stages when we only have text.
            image_path = ctx.document.image_path
            words_meta: list[dict[str, Any]] = []
            if self.render_image and not image_path and source_text:
                img, words_meta = render_page(source_text)
                out_dir = self.cfg.pipeline_cache_dir / "renders"
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_id = ctx.document.record_id.replace("::", "__").replace("/", "_")
                image_path = out_dir / f"{safe_id}.png"
                img.save(image_path)
                ctx.document.image_path = image_path
                labeled = label_words(words_meta)
                tokens = [w["text"] for w in labeled]
            else:
                tokens = source_text.split()

            if self._backend == "transformers" and tokens:
                fields = self._model_extract(tokens)
                backend = "transformers"
                # Fill gaps with heuristics so sparse smoke models still yield usable fields
                heuristic = _heuristic_extract(md_text)
                for k, v in heuristic.items():
                    if k not in fields or not fields[k]:
                        fields[k] = v
                        flags.append(f"extract_heuristic_fill:{k}")
                confidence = 0.7 if fields else 0.3
            else:
                fields = _heuristic_extract(md_text)
                backend = "heuristic"
                flags.append("extract_heuristic")
                confidence = 0.65 if fields else 0.25

            # Classification-conditioned soft prior: loss docs should have date_of_loss
            if doc_type == "loss_notice" and "date_of_loss" not in fields:
                flags.append("missing_expected_field:date_of_loss")
                confidence = min(confidence, 0.45)

            if confidence < LOW_CONFIDENCE:
                flags.append("low_confidence_extraction")

            flat = {k: (v[0] if v else None) for k, v in fields.items()}
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=True,
                confidence=confidence,
                flags=flags,
                payload={
                    "fields": fields,
                    "fields_flat": flat,
                    "document_type": doc_type,
                    "backend": backend,
                    "image_path": str(image_path) if image_path else None,
                    "n_fields": len(fields),
                    "input_from": "markdown" if ctx.markdown else "text",
                    "model_dir": str(self.model_dir) if self.model_dir else None,
                },
            )
        except Exception as exc:
            logger.exception("Extract stage failed")
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=False,
                confidence=0.0,
                flags=["extract_failed"],
                error=str(exc),
                payload={},
            )


@dataclass
class VisionLLMStage:
    """
    Local Vision-LLM / markdown-LLM stage that reacts to classify + extract.

    Prefer feeding upstream structured markdown (token-efficient). Optionally
    attach the page image when VISION_LLM_USE_IMAGE=1 for visual verification
    on high-RAM hosts. Without a configured model it runs a deterministic
    refine so the chain stays intact.
    """

    cfg: Config
    order: int = 3
    name: str = "vision_llm"
    enabled: bool = True
    _processor: Any = None
    _model: Any = None
    _backend: str = "unloaded"

    def _ensure_loaded(self) -> None:
        if self._backend != "unloaded":
            return
        if not self.enabled:
            self._backend = "disabled"
            return
        # Only attempt a real VLM download/load when a local path exists or
        # VISION_LLM_LOAD=1 is set — keeps CI / modest hosts on heuristic refine.
        model_id = self.cfg.vision_llm_model
        local_path = self.cfg.vision_llm_model_path
        load_remote = _bool_env("VISION_LLM_LOAD", default=False)
        if local_path and Path(local_path).exists():
            target: str | None = str(local_path)
        elif load_remote and model_id:
            target = model_id
        else:
            self._backend = "heuristic"
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(target, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                target,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                self._model = self._model.to("cpu")
            self._model.eval()
            self._torch = torch
            self._backend = "transformers_vlm"
            self._target = target
        except Exception as exc:
            logger.warning(
                "Vision LLM load failed for %s (%s); using heuristic refine",
                target,
                exc,
            )
            self._backend = "heuristic"

    def _vlm_refine(self, ctx: AnalysisContext) -> dict[str, Any]:
        assert self._model is not None and self._processor is not None
        prior_fields = (ctx.extraction or {}).get("fields_flat") or {}
        doc_type = (ctx.classification or {}).get("document_type", "unknown")
        md = ctx.content_for_llm()
        # Cap markdown fed to the model to keep generation focused
        md_excerpt = md if len(md) <= 6000 else md[:6000] + "\n\n…(truncated)…"
        prompt = (
            "You are an insurance document analyst. "
            f"Document type: {doc_type}. "
            f"Prior extracted fields: {prior_fields}.\n\n"
            "Document as markdown (preferred over raw page pixels for token efficiency):\n"
            f"{md_excerpt}\n\n"
            "Return a short JSON object of corrected key fields "
            "(claim_id, policy_number, policyholder_name, date_of_loss, estimated_damage, "
            "deductible, location). JSON only."
        )

        use_image = _bool_env("VISION_LLM_USE_IMAGE", default=False)
        image_path = ctx.document.image_path or (ctx.extraction or {}).get("image_path")
        if use_image and image_path:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self._processor(text=[text], images=[image], return_tensors="pt")
            except Exception:
                inputs = self._processor(images=image, text=prompt, return_tensors="pt")
            mode = "markdown+image"
        else:
            # Markdown-only path — avoids large vision token budgets
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ]
                text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self._processor(text=[text], return_tensors="pt")
            except Exception:
                inputs = self._processor(text=prompt, return_tensors="pt")
            mode = "markdown"

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=256)
        raw = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return {
            "raw_response": raw,
            "refined_fields": _parse_json_object(raw),
            "llm_input_mode": mode,
            "markdown_chars": len(md),
            "markdown_approx_tokens": approx_token_count(md),
        }

    def _heuristic_refine(self, ctx: AnalysisContext) -> dict[str, Any]:
        """
        Lightweight refine over markdown / plain text field tables.
        Keeps the chain reactive without requiring a heavy VLM download in CI.
        """
        fields = dict((ctx.extraction or {}).get("fields_flat") or {})
        text = ctx.content_for_llm() or ctx.document.text
        heuristic = {k: (v[0] if v else None) for k, v in _heuristic_extract(text).items()}
        refined = {**heuristic, **{k: v for k, v in fields.items() if v}}
        for k, v in heuristic.items():
            if not refined.get(k) and v:
                refined[k] = v
        notes = []
        image_path = ctx.document.image_path or (ctx.extraction or {}).get("image_path")
        if ctx.markdown:
            notes.append("markdown_context")
            notes.append(f"markdown_tokens≈{(ctx.markdown or {}).get('approx_tokens')}")
        if image_path:
            notes.append(f"image_present:{image_path}")
        else:
            notes.append("no_image")
        return {
            "refined_fields": refined,
            "notes": notes,
            "raw_response": None,
            "llm_input_mode": "markdown",
            "markdown_chars": len(text),
            "markdown_approx_tokens": approx_token_count(text),
        }

    def run(self, ctx: AnalysisContext) -> StageResult:
        if not self.enabled:
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=True,
                confidence=1.0,
                flags=["vision_llm_skipped"],
                payload={"skipped": True, "reason": "disabled"},
            )
        self._ensure_loaded()
        flags: list[str] = []
        try:
            if self._backend == "transformers_vlm":
                payload = self._vlm_refine(ctx)
                payload["backend"] = "transformers_vlm"
                payload["model"] = getattr(self, "_target", self.cfg.vision_llm_model)
                confidence = 0.75 if payload.get("refined_fields") else 0.4
            else:
                payload = self._heuristic_refine(ctx)
                payload["backend"] = "heuristic"
                payload["model"] = None
                flags.append("vision_llm_heuristic")
                confidence = 0.6 if payload.get("refined_fields") else 0.3

            # Merge refined fields back so summarize reacts to vision corrections
            refined = payload.get("refined_fields") or {}
            if ctx.extraction is not None and refined:
                flat = dict(ctx.extraction.get("fields_flat") or {})
                flat.update({k: v for k, v in refined.items() if v})
                ctx.extraction["fields_flat"] = flat
                nested = dict(ctx.extraction.get("fields") or {})
                for k, v in flat.items():
                    if v:
                        nested[k] = [v]
                ctx.extraction["fields"] = nested
                payload["merged_into_extraction"] = True

            if confidence < LOW_CONFIDENCE:
                flags.append("low_confidence_vision")

            return StageResult(
                stage=self.name,
                order=self.order,
                ok=True,
                confidence=confidence,
                flags=flags,
                payload=payload,
            )
        except Exception as exc:
            logger.exception("Vision LLM stage failed")
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=False,
                confidence=0.0,
                flags=["vision_llm_failed"],
                error=str(exc),
                payload={},
            )


def _bool_env(key: str, default: bool = False) -> bool:
    import os

    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_json_object(text: str) -> dict[str, Any]:
    import json

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass
class PredictOutcomeStage:
    """
    Predict expected claim disposition from upstream classify/extract/vision features.

    Gold `expected_outcome` on synthetic skeletons uses the same deterministic rule,
    so outcome accuracy is a predictive tracking metric alongside classification /
    extraction reports.
    """

    cfg: Config
    order: int = 4
    name: str = "predict_outcome"

    def run(self, ctx: AnalysisContext) -> StageResult:
        flags: list[str] = []
        try:
            flat = dict((ctx.extraction or {}).get("fields_flat") or {})
            if ctx.vision and ctx.vision.get("refined_fields"):
                flat.update({k: v for k, v in ctx.vision["refined_fields"].items() if v})

            meta = ctx.document.metadata or {}
            skeleton = meta.get("skeleton") if isinstance(meta.get("skeleton"), dict) else None
            if skeleton is None and isinstance(meta.get("expected_outcome"), str):
                skeleton = {"expected_outcome": meta["expected_outcome"]}

            # Prefer narrative complexity from metadata/skeleton when present.
            complexity = None
            if skeleton:
                complexity = skeleton.get("narrative_complexity")
            complexity = complexity or meta.get("narrative_complexity")

            prediction = predict_outcome(
                fields=flat,
                document_type=(ctx.classification or {}).get("document_type"),
                text=ctx.content_for_encoder(),
                narrative_complexity=complexity if isinstance(complexity, str) else None,
                gold_skeleton=skeleton,
            )

            confidence = float(prediction.get("confidence") or 0.0)
            if confidence < LOW_CONFIDENCE:
                flags.append("low_confidence_outcome")
            if prediction.get("features", {}).get("estimated_damage") is None:
                flags.append("outcome_missing_damage")
            if prediction.get("gold_outcome") is not None and not prediction.get("correct"):
                flags.append("outcome_mismatch_vs_gold")

            return StageResult(
                stage=self.name,
                order=self.order,
                ok=True,
                confidence=confidence,
                flags=flags,
                payload=prediction,
            )
        except Exception as exc:
            logger.exception("PredictOutcome stage failed")
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=False,
                confidence=0.0,
                flags=["predict_outcome_failed"],
                error=str(exc),
                payload={"label_set": list(OUTCOME_LABELS)},
            )


@dataclass
class SummarizeStage:
    """
    Memo generation — reacts to markdown, classification, extraction, vision,
    and predicted claim outcome.

    Uses a local generative model when configured; otherwise a deterministic
    template grounded only in upstream stage payloads (no skeleton peeking).
    LLM prompts receive structured markdown rather than raw page images.
    """

    cfg: Config
    order: int = 5
    name: str = "summarize"
    _model: Any = None
    _tokenizer: Any = None
    _backend: str = "unloaded"

    def _ensure_loaded(self) -> None:
        if self._backend != "unloaded":
            return
        path = self.cfg.summarizer_model_path or (self.cfg.models_dir / "summarizer")
        path = Path(path)
        if not (path / "config.json").exists() and not self.cfg.summarizer_model:
            self._backend = "template"
            return
        target = str(path) if (path / "config.json").exists() else self.cfg.summarizer_model
        if not target:
            self._backend = "template"
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(target)
            self._model = AutoModelForCausalLM.from_pretrained(target)
            self._model.eval()
            self._torch = torch
            self._backend = "transformers"
            self._target = target
        except Exception as exc:
            logger.warning("Summarizer load failed (%s); using template", exc)
            self._backend = "template"

    def _template_memo(self, ctx: AnalysisContext) -> str:
        clf = ctx.classification or {}
        ext = ctx.extraction or {}
        vision = ctx.vision or {}
        outcome = ctx.outcome or {}
        md_meta = ctx.markdown or {}
        flat = dict(ext.get("fields_flat") or {})
        if vision.get("refined_fields"):
            flat.update({k: v for k, v in vision["refined_fields"].items() if v})

        claim_id = flat.get("claim_id") or ctx.document.claim_id or ctx.document.record_id
        adjuster = flat.get("adjuster_assigned") or "Unassigned Adjuster"
        insured = flat.get("policyholder_name") or "Unknown Insured"
        policy = flat.get("policy_number") or "Unknown Policy"
        loss_type = flat.get("loss_type") or "unspecified"
        date_of_loss = flat.get("date_of_loss") or "unknown date"
        location = flat.get("location") or "unspecified location"
        damage = flat.get("estimated_damage") or "n/a"
        deductible = flat.get("deductible") or "n/a"
        reserve = flat.get("reserve_set") or "n/a"
        coverage = flat.get("coverage_type") or "unspecified coverage"
        state = flat.get("state") or ""
        doc_type = clf.get("document_type") or "unknown"
        predicted = (
            outcome.get("expected_outcome")
            or outcome.get("outcome_label")
            or "unspecified"
        )
        outcome_conf = outcome.get("confidence", "n/a")
        outcome_desc = outcome.get("description") or ""

        flags = list(dict.fromkeys(ctx.flags))
        review = "Yes — low-confidence upstream stage(s)" if any(
            f.startswith("low_confidence") for f in flags
        ) else "No"

        return "\n".join(
            [
                f"ADJUSTER MEMO — {claim_id}",
                "To: Claims File",
                f"From: {adjuster}",
                f"Re: {insured} / {policy}",
                "",
                "Summary",
                f"Based on automated chained analysis of a `{doc_type}` document, "
                f"the reported {loss_type} loss on {date_of_loss} under {coverage}"
                + (f" in {state}" if state else "")
                + ".",
                "",
                "Facts",
                f"- Location: {location}",
                f"- Estimated damage: {damage}",
                f"- Deductible: {deductible}",
                f"- Current reserve: {reserve}",
                f"- Source classification confidence: {clf.get('confidence', 'n/a')}",
                f"- Extraction backend: {ext.get('backend', 'n/a')}",
                f"- Vision backend: {(vision or {}).get('backend', 'skipped')}",
                f"- Predicted claim outcome: `{predicted}` "
                f"(confidence {outcome_conf})"
                + (f" — {outcome_desc}" if outcome_desc else ""),
                f"- Markdown backend: {md_meta.get('backend', 'n/a')} "
                f"(~{md_meta.get('approx_tokens', 'n/a')} tokens; "
                f"saved ≈{md_meta.get('token_saved_est', 'n/a')})",
                "",
                "Analysis",
                "Issue: whether coverage appears supported by extracted claim facts. "
                "Rule: coverage turns on the policy declarations, conditions, and applicable exclusions. "
                f"Application: extracted fields from the inbound document at {location}. "
                f"Conclusion: predicted disposition `{predicted}`; proceed with investigation "
                "and reserve adequacy review pending human confirmation of low-confidence fields.",
                "",
                "Next Steps",
                "- Confirm coverage grant/denial points in writing",
                "- Update reserve if investigation changes exposure",
                "- Request any missing supporting evidence",
                f"- Human review required: {review}",
                "",
                f"Pipeline flags: {', '.join(flags) if flags else 'none'}",
            ]
        )

    def _model_memo(self, ctx: AnalysisContext) -> str:
        assert self._model is not None and self._tokenizer is not None
        flat = dict((ctx.extraction or {}).get("fields_flat") or {})
        if ctx.vision and ctx.vision.get("refined_fields"):
            flat.update({k: v for k, v in ctx.vision["refined_fields"].items() if v})
        md = ctx.content_for_llm()
        md_excerpt = md if len(md) <= 4000 else md[:4000] + "\n\n…(truncated)…"
        prompt = (
            "Write a concise insurance adjuster memo grounded only in the markdown "
            "document and extracted fields. Do not invent claim details.\n"
            f"Document type: {(ctx.classification or {}).get('document_type')}\n"
            f"Fields: {flat}\n\n"
            f"Document markdown:\n{md_excerpt}\n\n"
            "Memo:\n"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with self._torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=400)
        return self._tokenizer.decode(out[0], skip_special_tokens=True)

    def run(self, ctx: AnalysisContext) -> StageResult:
        self._ensure_loaded()
        flags: list[str] = []
        try:
            # Require prior stages to have run (chronological reaction)
            if ctx.markdown is None:
                flags.append("summarize_missing_markdown")
            if ctx.classification is None:
                flags.append("summarize_missing_classification")
            if ctx.extraction is None:
                flags.append("summarize_missing_extraction")
            if ctx.outcome is None:
                flags.append("summarize_missing_outcome")

            if self._backend == "transformers":
                try:
                    memo = self._model_memo(ctx)
                    backend = "transformers"
                except Exception as exc:
                    logger.warning("Generative summarize failed (%s); template", exc)
                    memo = self._template_memo(ctx)
                    backend = "template_fallback"
                    flags.append("summarize_template_fallback")
            else:
                memo = self._template_memo(ctx)
                backend = "template"

            confidence = 0.8 if (ctx.extraction or {}).get("fields_flat") else 0.4
            if any(f.startswith("low_confidence") for f in ctx.flags):
                confidence = min(confidence, 0.5)
                flags.append("summarize_upstream_uncertainty")

            return StageResult(
                stage=self.name,
                order=self.order,
                ok=True,
                confidence=confidence,
                flags=flags,
                payload={
                    "memo": memo,
                    "backend": backend,
                    "input_from": "markdown",
                    "markdown_approx_tokens": (ctx.markdown or {}).get("approx_tokens"),
                    "grounded_in": [
                        s.stage for s in ctx.stages if s.ok and s.stage != self.name
                    ],
                },
            )
        except Exception as exc:
            logger.exception("Summarize stage failed")
            return StageResult(
                stage=self.name,
                order=self.order,
                ok=False,
                confidence=0.0,
                flags=["summarize_failed"],
                error=str(exc),
                payload={},
            )
