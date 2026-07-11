"""Claim outcome labels and predictive rules.

`expected_outcome` is a synthetic supervisory label for tracking how well the
pipeline's extracted claim features support a downstream settlement prediction.
Gold labels are a deterministic function of skeleton features so evaluation
accuracy measures end-to-end feature recovery + decision rules (not random noise).
"""

from __future__ import annotations

from typing import Any

OUTCOME_LABELS: tuple[str, ...] = (
    "pay_full",
    "pay_partial",
    "deny",
    "investigate",
    "close_without_payment",
)

OUTCOME_DESCRIPTIONS: dict[str, str] = {
    "pay_full": "Approve and pay the claim at (or near) presented exposure",
    "pay_partial": "Settle for less than claimed / apply limits or comparative fault",
    "deny": "Deny coverage (exclusion, condition breach, or fraud indicators)",
    "investigate": "Hold decision pending further investigation / SIU / medical review",
    "close_without_payment": "Close with no indemnity (below deductible / withdrawn / no coverage trigger)",
}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "")
    try:
        return float(text)
    except ValueError:
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lower = str(value).strip().lower()
    if lower in {"1", "true", "yes", "y", "filed", "reported"}:
        return True
    if lower in {"0", "false", "no", "n", "none", "not filed", "not reported"}:
        return False
    return None


def features_from_skeleton(skeleton: dict[str, Any]) -> dict[str, Any]:
    """Flatten skeleton fields used by the outcome decision rule."""
    loss = skeleton.get("loss_event") or {}
    fin = skeleton.get("financials") or {}
    return {
        "narrative_complexity": skeleton.get("narrative_complexity"),
        "loss_type": loss.get("loss_type"),
        "injuries_reported": loss.get("injuries_reported"),
        "police_report_filed": loss.get("police_report_filed"),
        "estimated_damage": fin.get("estimated_damage"),
        "deductible": fin.get("deductible"),
        "reserve_set": fin.get("reserve_set"),
        "document_type": skeleton.get("document_type"),
    }


def features_from_extraction(
    *,
    fields: dict[str, Any] | None,
    document_type: str | None = None,
    text: str = "",
    narrative_complexity: str | None = None,
) -> dict[str, Any]:
    """Build predictor features from upstream pipeline payloads + text cues."""
    flat = dict(fields or {})
    lower = (text or "").lower()

    injuries = _as_bool(flat.get("injuries_reported"))
    if injuries is None:
        injuries = any(
            k in lower
            for k in ("injuries reported", "injury reported", "bodily injury", "injured")
        ) and "no injuries" not in lower and "injuries: none" not in lower

    police = _as_bool(flat.get("police_report_filed"))
    if police is None:
        police = "police report" in lower and "no police" not in lower

    complexity = narrative_complexity
    if not complexity:
        if any(k in lower for k in ("fraud", "siu", "misrepresentation", "red flag")):
            complexity = "fraud_flagged"
        elif any(k in lower for k in ("ambiguous", "unclear liability", "conflicting")):
            complexity = "ambiguous"
        elif any(k in lower for k in ("clean loss", "straightforward", "no complications")):
            complexity = "clean"
        else:
            complexity = "standard"

    return {
        "narrative_complexity": complexity,
        "loss_type": flat.get("loss_type"),
        "injuries_reported": injuries,
        "police_report_filed": police,
        "estimated_damage": _as_float(flat.get("estimated_damage")),
        "deductible": _as_float(flat.get("deductible")),
        "reserve_set": _as_float(flat.get("reserve_set")),
        "document_type": document_type or flat.get("document_type"),
    }


def derive_expected_outcome(features: dict[str, Any]) -> str:
    """
    Deterministic claim-outcome label from claim features.

    Used both when sampling synthetic skeletons (gold label) and as the core
    decision rule inside the predict_outcome stage (prediction).
    """
    complexity = str(features.get("narrative_complexity") or "standard")
    injuries = bool(features.get("injuries_reported"))
    damage = _as_float(features.get("estimated_damage"))
    deductible = _as_float(features.get("deductible"))
    reserve = _as_float(features.get("reserve_set"))

    if complexity == "fraud_flagged":
        if damage is not None and damage >= 10_000:
            return "deny"
        return "investigate"

    if complexity == "ambiguous" or injuries:
        return "investigate"

    if damage is not None and deductible is not None and damage <= deductible:
        return "close_without_payment"

    if complexity == "clean" and (damage is None or damage < 10_000):
        return "pay_full"

    if damage is not None and damage >= 50_000:
        return "pay_partial"

    if (
        damage is not None
        and reserve is not None
        and reserve > 0
        and damage / reserve >= 1.35
    ):
        return "investigate"

    if complexity == "standard" and damage is not None and damage >= 15_000:
        return "pay_partial"

    if damage is None and deductible is None:
        # Underwriting / certificate docs often lack loss financials.
        doc = str(features.get("document_type") or "")
        if doc.startswith("application") or doc in {
            "certificate_evidence",
            "policy_change_endorsement",
        }:
            return "investigate"
        return "investigate"

    return "pay_full"


def predict_outcome(
    *,
    fields: dict[str, Any] | None = None,
    document_type: str | None = None,
    text: str = "",
    narrative_complexity: str | None = None,
    gold_skeleton: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Predict expected claim outcome from pipeline features.

    Returns label, confidence, feature snapshot, and optional gold comparison
    when a skeleton with `expected_outcome` (or derivable features) is supplied.
    """
    features = features_from_extraction(
        fields=fields,
        document_type=document_type,
        text=text,
        narrative_complexity=narrative_complexity,
    )
    label = derive_expected_outcome(features)

    # Confidence: higher when key financials are present and complexity is sharp.
    present = sum(
        1
        for k in ("estimated_damage", "deductible", "loss_type", "narrative_complexity")
        if features.get(k) not in (None, "")
    )
    confidence = 0.35 + 0.12 * present
    if features.get("narrative_complexity") in {"fraud_flagged", "ambiguous", "clean"}:
        confidence += 0.1
    if features.get("estimated_damage") is None:
        confidence = min(confidence, 0.55)
    confidence = float(min(0.95, confidence))

    payload: dict[str, Any] = {
        "expected_outcome": label,
        "outcome_label": label,
        "confidence": confidence,
        "features": features,
        "label_set": list(OUTCOME_LABELS),
        "description": OUTCOME_DESCRIPTIONS.get(label, ""),
        "backend": "deterministic_rules",
    }

    if gold_skeleton:
        gold = gold_skeleton.get("expected_outcome")
        if not gold:
            gold = derive_expected_outcome(features_from_skeleton(gold_skeleton))
        payload["gold_outcome"] = gold
        payload["correct"] = gold == label

    return payload
