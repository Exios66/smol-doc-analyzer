"""Application profiles for medical bills and salvage claims (paper §III)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import REPO_ROOT

TAXONOMY_DIR = REPO_ROOT / "taxonomy"

APPLICATION_TAXONOMIES = {
    "medical_bills": TAXONOMY_DIR / "medical_bills.yaml",
    "salvage_claims": TAXONOMY_DIR / "salvage_claims.yaml",
    # Existing ACORD taxonomy reused when callers want the general intake set.
    "acord": TAXONOMY_DIR / "acord_form_categories.yaml",
}


@dataclass(frozen=True)
class ApplicationProfile:
    name: str
    description: str
    labels: list[str]
    aliases: dict[str, list[str]]
    extraction_fields: list[str]
    prefer_non_other: bool = True
    review_confidence_threshold: float = 0.55
    taxonomy_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def label_set(self) -> set[str]:
        return set(self.labels)


def load_application(name: str) -> ApplicationProfile:
    """Load a DICIE application profile from taxonomy YAML."""
    key = name.strip().lower().replace("-", "_")
    if key not in APPLICATION_TAXONOMIES:
        known = ", ".join(sorted(APPLICATION_TAXONOMIES))
        raise ValueError(f"Unknown application {name!r}. Choose one of: {known}")

    path = APPLICATION_TAXONOMIES[key]
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    categories = raw.get("categories") or []
    labels: list[str] = []
    aliases: dict[str, list[str]] = {}
    for cat in categories:
        label = str(cat["label"]).lower()
        labels.append(label)
        alias_list = [str(a) for a in (cat.get("aliases") or [])]
        aliases[label] = alias_list

    rules = raw.get("business_rules") or {}
    fields = [str(f) for f in (raw.get("extraction_fields") or [])]
    if key == "acord" and not fields:
        # ACORD intake reuses the synthetic form field set.
        fields = [
            "claim_id",
            "policy_number",
            "policyholder_name",
            "date_of_loss",
            "loss_type",
            "location",
            "estimated_damage",
            "deductible",
        ]

    return ApplicationProfile(
        name=str(raw.get("application") or key),
        description=str(raw.get("description") or "").strip(),
        labels=labels,
        aliases=aliases,
        extraction_fields=fields,
        prefer_non_other=bool(rules.get("prefer_non_other", True)),
        review_confidence_threshold=float(rules.get("review_confidence_threshold", 0.55)),
        taxonomy_path=path,
        raw=raw,
    )


def list_applications() -> list[str]:
    return sorted(APPLICATION_TAXONOMIES)
