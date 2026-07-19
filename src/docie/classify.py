"""
Stage 2 — Document Classification (paper §IV.A / Fig. 1).

Classify each page image (text + optional visual features) and aggregate
page-level predictions into a single document label.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from src.docie.applications import ApplicationProfile
from src.docie.types import ClassificationResult, PageClassification, ProcessedDocument

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _keyword_scores(text: str, profile: ApplicationProfile) -> dict[str, float]:
    """Lightweight keyword / alias scoring per application class."""
    lower = _normalize(text)
    scores = {label: 0.0 for label in profile.labels}
    if not lower:
        return scores

    # Application-specific cue phrases (paper §III samples).
    cues: dict[str, list[str]] = {
        # medical
        "hcfa": [
            "hcfa",
            "cms-1500",
            "cms 1500",
            "health care finance",
            "physician or supplier",
            "carrier name",
        ],
        "ub04": [
            "ub-04",
            "ub04",
            "ub 04",
            "uniform billing",
            "cms-1450",
            "type of bill",
            "revenue code",
        ],
        # salvage
        "log": [
            "letter of guarantee",
            "letter of guaranty",
            "guarantee that",
            "lienholder",
            "payoff",
            "guarantees that the insurer",
        ],
        "sales": [
            "sales receipt",
            "sales tax",
            "bill of sale",
            "purchase price",
            "sold to",
            "buyer",
        ],
        "other": [],
    }

    for label in profile.labels:
        hits = 0
        candidates = list(profile.aliases.get(label, []))
        candidates.extend(cues.get(label, []))
        # Also score the label token itself when distinctive.
        if label not in {"other"}:
            candidates.append(label.replace("_", " "))
        for cue in candidates:
            cue_l = cue.lower()
            if cue_l and cue_l in lower:
                hits += 1
        if candidates:
            scores[label] = hits / max(len(set(c.lower() for c in candidates)), 1)
        else:
            scores[label] = 0.0

    # Soft prior for "other" when nothing else fires.
    if all(v <= 0 for k, v in scores.items() if k != "other") and "other" in scores:
        scores["other"] = 0.35
    return scores


def classify_page_text(
    text: str,
    profile: ApplicationProfile,
    *,
    page_index: int = 0,
) -> PageClassification:
    scores = _keyword_scores(text, profile)
    best = max(scores, key=scores.get)
    conf = float(scores[best])
    if conf <= 0:
        best = "other" if "other" in scores else profile.labels[0]
        conf = 0.2
    else:
        conf = min(0.95, 0.4 + conf * 0.6)
    return PageClassification(
        page_index=page_index,
        label=best,
        confidence=conf,
        backend="heuristic_text",
        scores=scores,
    )


def _try_vit_classify(
    image_path: str,
    profile: ApplicationProfile,
    model_dir: Any,
) -> PageClassification | None:
    """Optional ViT image classifier when weights match the label set."""
    from pathlib import Path

    path = Path(model_dir) if model_dir else None
    if path is None or not (path / "config.json").exists():
        return None
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from PIL import Image

        from src.utils.io import read_json

        processor = AutoImageProcessor.from_pretrained(str(path))
        model = AutoModelForImageClassification.from_pretrained(str(path))
        model.eval()
        label2id = read_json(path / "label2id.json")
        id2label = {int(v): str(k).lower() for k, v in label2id.items()}
        # Only use ViT when its labels intersect the application taxonomy.
        if not set(id2label.values()) & profile.label_set:
            return None
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(probs).item())
            conf = float(probs[pred_id].item())
        label = id2label.get(pred_id, profile.labels[0])
        if label not in profile.label_set:
            # Map unknown label → other / first
            label = "other" if "other" in profile.label_set else profile.labels[0]
        scores = {
            id2label.get(i, f"cls_{i}"): float(probs[i].item())
            for i in range(probs.shape[0])
        }
        return PageClassification(
            page_index=0,
            label=label,
            confidence=conf,
            backend="vit",
            scores=scores,
        )
    except Exception as exc:
        logger.warning("ViT classify unavailable: %s", exc)
        return None


def aggregate_page_predictions(
    pages: list[PageClassification],
    profile: ApplicationProfile,
) -> ClassificationResult:
    """
    Aggregate page-level class labels into a document prediction.

    Uses confidence-weighted majority vote. When prefer_non_other is set,
    ties between `other` and a specific class resolve to the specific class.
    """
    if not pages:
        fallback = "other" if "other" in profile.label_set else profile.labels[0]
        return ClassificationResult(
            label=fallback,
            confidence=0.0,
            backend="empty",
            page_predictions=[],
            aggregation="majority_vote",
            flags=["no_pages"],
        )

    weights: Counter[str] = Counter()
    conf_sum: dict[str, float] = {lab: 0.0 for lab in profile.labels}
    for pred in pages:
        label = pred.label if pred.label in profile.label_set else (
            "other" if "other" in profile.label_set else profile.labels[0]
        )
        weights[label] += pred.confidence
        conf_sum[label] = conf_sum.get(label, 0.0) + pred.confidence

    # Rank by weighted votes
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_weight = ranked[0]
    if (
        profile.prefer_non_other
        and best_label == "other"
        and len(ranked) > 1
        and abs(ranked[0][1] - ranked[1][1]) < 1e-6
    ):
        best_label = ranked[1][0]
        best_weight = ranked[1][1]

    # Soft prefer non-other when it is close (within 15%) of other
    if profile.prefer_non_other and best_label == "other" and len(ranked) > 1:
        alt_label, alt_weight = ranked[1]
        if alt_label != "other" and alt_weight >= 0.85 * best_weight:
            best_label, best_weight = alt_label, alt_weight

    total = sum(weights.values()) or 1.0
    confidence = float(best_weight / total)
    # Blend with mean page confidence for the winning label
    matching = [p.confidence for p in pages if p.label == best_label]
    if matching:
        confidence = 0.5 * confidence + 0.5 * (sum(matching) / len(matching))

    backends = sorted({p.backend for p in pages})
    flags: list[str] = []
    if confidence < profile.review_confidence_threshold:
        flags.append("low_confidence_classification")

    return ClassificationResult(
        label=best_label,
        confidence=min(0.99, confidence),
        backend="+".join(backends),
        page_predictions=pages,
        aggregation="confidence_weighted_majority",
        flags=flags,
    )


def classify_document(
    processed: ProcessedDocument,
    profile: ApplicationProfile,
    *,
    vit_model_dir: Any = None,
) -> ClassificationResult:
    """Stage 2: classify each page, then aggregate to a document label."""
    page_preds: list[PageClassification] = []
    for page in processed.pages:
        vit_pred = None
        if vit_model_dir is not None:
            vit_pred = _try_vit_classify(
                str(page.image_path), profile, vit_model_dir
            )
            if vit_pred is not None:
                vit_pred.page_index = page.page_index

        text_pred = classify_page_text(
            page.text, profile, page_index=page.page_index
        )

        if vit_pred is not None:
            # Blend ViT + text when both available
            blended_scores = dict(text_pred.scores)
            for k, v in vit_pred.scores.items():
                if k in blended_scores:
                    blended_scores[k] = 0.5 * blended_scores[k] + 0.5 * v
                elif k in profile.label_set:
                    blended_scores[k] = v
            best = max(
                (lab for lab in profile.labels),
                key=lambda lab: blended_scores.get(lab, 0.0),
            )
            conf = float(blended_scores.get(best, 0.0))
            conf = min(0.95, 0.4 + conf * 0.6) if conf > 0 else max(
                text_pred.confidence, vit_pred.confidence
            )
            page_preds.append(
                PageClassification(
                    page_index=page.page_index,
                    label=best,
                    confidence=conf,
                    backend="vit+heuristic_text",
                    scores=blended_scores,
                )
            )
        else:
            page_preds.append(text_pred)

    return aggregate_page_predictions(page_preds, profile)
