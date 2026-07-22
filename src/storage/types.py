"""Dataclasses for the sample document corpus store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ClaimRecord:
    """Claim-level container that can own multiple related documents."""

    claim_id: str
    application: str
    carrier_name: str
    state: str | None = None
    date_of_loss: str | None = None
    loss_type: str | None = None
    policy_number: str | None = None
    insured_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FieldRecord:
    """A single labeled field attached to a document."""

    field_name: str
    field_value: str | None
    field_role: str = "ground_truth"  # ground_truth | extracted | annotation
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentRecord:
    """Canonical stored document with optional structured skeleton + labels."""

    document_id: str
    application: str
    document_type: str
    text: str
    claim_id: str | None = None
    title: str | None = None
    source_kind: str = "synthetic_seed"
    is_synthetic: bool = True
    split: str | None = None  # train | val | test | None
    source_path: str | None = None
    skeleton: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: list[FieldRecord] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def ground_truth_fields(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for f in self.fields:
            if f.field_role == "ground_truth":
                out[f.field_name] = f.field_value
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_docie_row(self) -> dict[str, Any]:
        """Export shape compatible with DICIE JSONL inputs / eval gold sets."""
        row: dict[str, Any] = {
            "record_id": self.document_id,
            "application": self.application,
            "document_type": self.document_type,
            "text": self.text,
            "ground_truth_fields": self.ground_truth_fields(),
        }
        if self.claim_id:
            row["claim_id"] = self.claim_id
        if self.split:
            row["split"] = self.split
        if self.source_path:
            row["source_path"] = self.source_path
        if self.skeleton:
            row["skeleton"] = self.skeleton
        if self.metadata:
            row["metadata"] = self.metadata
        return row

    def to_classification_row(self) -> dict[str, Any]:
        return {
            "record_id": self.document_id,
            "text": self.text,
            "label": self.document_type,
            "application": self.application,
            "claim_id": self.claim_id,
            "split": self.split,
        }

    def to_extraction_row(self) -> dict[str, Any]:
        return {
            "record_id": self.document_id,
            "text": self.text,
            "document_type": self.document_type,
            "application": self.application,
            "claim_id": self.claim_id,
            "fields": self.ground_truth_fields(),
            "split": self.split,
        }
