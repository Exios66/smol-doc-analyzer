"""
Stage 3 — Information Extraction (paper §IV.B / Fig. 1).

Extract application-specific fields from OCR text / word boxes, conditioned
on the Stage 2 document classification. Uses LayoutLM-style token
classification when weights are available; otherwise regex / pattern
heuristics aligned with the medical and salvage field sets.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.docie.applications import ApplicationProfile
from src.docie.types import ExtractionResult, ProcessedDocument

logger = logging.getLogger(__name__)

# Field patterns for paper applications + shared claim identifiers.
FIELD_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "claim_id": [
        re.compile(
            r"\bclaim\s*(?:number|no\.?|id|#)\s*[:#]\s*([A-Z0-9][A-Z0-9\-_/]{4,})",
            re.I,
        ),
        re.compile(r"\b(CLM[-\s]?\d{4}[-\s]?\d+)\b", re.I),
    ],
    "name": [
        re.compile(
            r"\b(?:patient\s*name|insured\s*name|claimant\s*name)\s*[:#]\s*([^\n]+)",
            re.I,
        ),
        # Prefer "Patient:" style labels; bare "Name:" must not match "Carrier Name:".
        re.compile(
            r"\bpatient\s*[:#]\s*([^\n]+)",
            re.I,
        ),
        re.compile(
            r"(?<!\bcarrier\s)(?<!\binsured\s)(?<!\bclaimant\s)\bname\s*[:#]\s*([^\n]+)",
            re.I,
        ),
    ],
    "dob": [
        re.compile(
            r"\b(?:date\s*of\s*birth|dob|birth\s*date)\s*[:#]?\s*"
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.I,
        ),
    ],
    "patient_id": [
        re.compile(
            r"\b(?:patient\s*(?:identifier|account|id)|pid|mrn)\b\s*[:#]?\s*"
            r"([A-Z0-9][A-Z0-9\-_/]{3,})",
            re.I,
        ),
    ],
    "address": [
        re.compile(
            r"\b(?:address|street\s*address|patient\s*address)\s*[:#]?\s*"
            r"(.+?)(?:\n|$)",
            re.I,
        ),
    ],
    "vin": [
        re.compile(
            r"\b(?:vin|vehicle\s*identification\s*(?:number|no\.?)?)\s*[:#]?\s*"
            r"([A-HJ-NPR-Z0-9]{11,17})\b",
            re.I,
        ),
        re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b"),
    ],
    "year": [
        re.compile(
            r"\b(?:year|model\s*year|yr)\s*[:#]?\s*((?:19|20)\d{2})\b",
            re.I,
        ),
        # "Vehicle: 2015 Ford Focus"
        re.compile(
            r"\bvehicle\s*[:#]\s*((?:19|20)\d{2})\s+[A-Za-z]",
            re.I,
        ),
    ],
    "make": [
        re.compile(
            r"\b(?:make|manufacturer)\s*[:#]?\s*([A-Za-z][A-Za-z0-9\- ]{1,20})",
            re.I,
        ),
        re.compile(
            r"\bvehicle\s*[:#]\s*(?:19|20)\d{2}\s+([A-Za-z][A-Za-z0-9\-]*)",
            re.I,
        ),
    ],
    "model": [
        re.compile(
            r"\b(?:model)\s*[:#]?\s*([A-Za-z0-9][A-Za-z0-9\- ]{1,30})",
            re.I,
        ),
        re.compile(
            r"\bvehicle\s*[:#]\s*(?:19|20)\d{2}\s+[A-Za-z][A-Za-z0-9\-]*\s+"
            r"([A-Za-z0-9][A-Za-z0-9\- ]{1,30})",
            re.I,
        ),
    ],
    "policy_number": [
        re.compile(
            r"\b(?:policy\s*(?:number|no\.?|#)?)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_/]{4,})",
            re.I,
        ),
    ],
    "policyholder_name": [
        re.compile(
            r"\b(?:policyholder|named\s*insured|insured)\s*[:#]?\s*"
            r"([A-Z][A-Za-z'`\-\.]+(?:\s+[A-Z][A-Za-z'`\-\.]+){0,3})",
            re.I,
        ),
    ],
    "date_of_loss": [
        re.compile(
            r"\b(?:date\s*of\s*loss|loss\s*date|dol)\s*[:#]?\s*"
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
            re.I,
        ),
    ],
    "loss_type": [
        re.compile(r"\b(?:loss\s*type|type\s*of\s*loss)\s*[:#]?\s*([A-Za-z][\w \-/]{2,40})", re.I),
    ],
    "location": [
        re.compile(
            r"\b(?:loss\s*location|location|place\s*of\s*loss)\s*[:#]?\s*(.+?)(?:\n|$)",
            re.I,
        ),
    ],
    "estimated_damage": [
        re.compile(
            r"\b(?:estimated\s*damage|estimate\s*total|damage\s*amount)\s*[:#]?\s*"
            r"(\$?[\d,]+\.?\d*)",
            re.I,
        ),
    ],
    "deductible": [
        re.compile(r"\b(?:deductible)\s*[:#]?\s*(\$?[\d,]+\.?\d*)", re.I),
    ],
}


# Require whitespace before the next label so values like "Model 3" survive.
_NEXT_FIELD_CUT = re.compile(
    r"\s+\b(?:claim\s*(?:number|no\.?|id)|vin|make|model|year|name|dob|"
    r"date(?:\s+of\s+birth)|patient\s*(?:id|identifier|account)|address|"
    r"carrier|physician|type\s*of\s*bill|revenue\s*code|payoff|"
    r"purchase\s*price|sales\s*tax|sold\s*to|buyer|amount\s*due|"
    r"account|balance)\b.*$",
    re.I,
)

_NAME_STOP = re.compile(
    r"^(?P<name>[A-Z][A-Za-z'`\-\.]+(?:\s+[A-Z][A-Za-z'`\-\.]*){0,3})\b",
)


def _clean_value(value: str) -> str:
    value = value.strip().strip(".,;: ")
    value = _NEXT_FIELD_CUT.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clean_name(value: str) -> str:
    value = _clean_value(value)
    m = _NAME_STOP.match(value)
    if m:
        return m.group("name").strip()
    # Fallback: first 1–4 capitalized tokens
    parts = value.split()
    kept: list[str] = []
    for part in parts[:4]:
        if re.match(r"^[A-Z][A-Za-z'`\-\.]*$", part):
            kept.append(part)
        else:
            break
    return " ".join(kept) if kept else value


def heuristic_extract(text: str, fields: list[str]) -> dict[str, list[str]]:
    """Regex extraction restricted to the application field set."""
    out: dict[str, list[str]] = {}
    for field in fields:
        patterns = FIELD_PATTERNS.get(field) or []
        found: list[str] = []
        for pat in patterns:
            for match in pat.finditer(text or ""):
                val = _clean_value(match.group(1))
                if not val:
                    continue
                # Avoid capturing a bare field label as a value (allow "Model 3")
                if re.match(
                    r"^(claim|vin|make|model|year|name|dob|address|patient)$",
                    val,
                    re.I,
                ):
                    continue
                if field == "name":
                    val = _clean_name(val)
                elif field in {"make", "model"}:
                    val = re.split(
                        r"\s+(?:Claim|VIN|Year|Make|Model|DOB|Address|Carrier|PID|MRN)\b",
                        val,
                        maxsplit=1,
                        flags=re.I,
                    )[0].strip()
                if val and val not in found:
                    found.append(val)
        if found:
            out[field] = found
    return out


def _try_layoutlm_extract(
    processed: ProcessedDocument,
    profile: ApplicationProfile,
    model_dir: Any,
) -> ExtractionResult | None:
    """Optional LayoutLM / token-classifier path when weights exist."""
    from pathlib import Path

    path = Path(model_dir) if model_dir else None
    if path is None or not (path / "config.json").exists():
        return None
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        from src.extraction.eval import _decode_entities
        from src.utils.io import read_json

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForTokenClassification.from_pretrained(str(path))
        model.eval()
        label2id = {k: int(v) for k, v in read_json(path / "label2id.json").items()}
        id2label = {v: k for k, v in label2id.items()}

        # Prefer the page with the most OCR words
        page = max(processed.pages, key=lambda p: len(p.words) or len(p.text.split()), default=None)
        if page is None:
            return None
        tokens = [w.text for w in page.words] if page.words else page.text.split()
        if not tokens:
            return None

        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**enc).logits[0]
            pred_ids = torch.argmax(logits, dim=-1).tolist()
        word_ids = enc.word_ids(batch_index=0)
        aligned = ["O"] * len(tokens)
        seen: set[int] = set()
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            aligned[wid] = id2label.get(int(pred_ids[idx]), "O")
        decoded = dict(_decode_entities(tokens, aligned))
        # Keep only application fields (case-insensitive key match)
        wanted = {f.lower() for f in profile.extraction_fields}
        fields: dict[str, list[str]] = {}
        for k, vals in decoded.items():
            key = k.lower()
            if key in wanted and vals:
                fields[key] = list(vals)

        flat = {f: (fields[f][0] if f in fields and fields[f] else None) for f in profile.extraction_fields}
        n = sum(1 for v in flat.values() if v)
        return ExtractionResult(
            fields=fields,
            fields_flat=flat,
            backend="layoutlm",
            confidence=0.75 if n else 0.3,
            page_index=page.page_index,
        )
    except Exception as exc:
        logger.warning("LayoutLM extract unavailable: %s", exc)
        return None


def extract_information(
    processed: ProcessedDocument,
    profile: ApplicationProfile,
    *,
    document_type: str | None = None,
    extractor_dir: Any = None,
) -> ExtractionResult:
    """
    Stage 3: extract fields from processed page images / OCR text.

    Extraction is conditioned on classification when that changes which
    fields are expected (e.g. LOG vs OTHER in salvage).
    """
    flags: list[str] = []
    text = processed.full_text
    model_result = _try_layoutlm_extract(processed, profile, extractor_dir)

    heuristic = heuristic_extract(text, profile.extraction_fields)
    if model_result is not None:
        fields = dict(model_result.fields)
        fill = 0
        for k, vals in heuristic.items():
            if k not in fields or not fields[k]:
                fields[k] = vals
                fill += 1
                flags.append(f"extract_heuristic_fill:{k}")
        backend = "layoutlm+heuristic" if fill else "layoutlm"
        flat = {
            f: (fields[f][0] if f in fields and fields[f] else None)
            for f in profile.extraction_fields
        }
        n = sum(1 for v in flat.values() if v)
        confidence = 0.7 if n else 0.3
        if fill and n:
            confidence = min(confidence, 0.55 + 0.05 * n)
        page_index = model_result.page_index
    else:
        fields = heuristic
        backend = "heuristic"
        flags.append("extract_heuristic")
        flat = {
            f: (fields[f][0] if f in fields and fields[f] else None)
            for f in profile.extraction_fields
        }
        n = sum(1 for v in flat.values() if v)
        confidence = min(0.85, 0.35 + 0.1 * n) if n else 0.25
        page_index = 0 if processed.pages else None

    # Classification-conditioned expectations (paper applications).
    doc_type = (document_type or "").lower()
    if profile.name == "salvage_claims" and doc_type == "log":
        for required in ("claim_id", "vin"):
            if not flat.get(required):
                flags.append(f"missing_expected_field:{required}")
                confidence = min(confidence, 0.45)
    if profile.name == "medical_bills" and doc_type in {"hcfa", "ub04"}:
        for required in ("claim_id", "name"):
            if not flat.get(required):
                flags.append(f"missing_expected_field:{required}")
                confidence = min(confidence, 0.45)

    if confidence < profile.review_confidence_threshold:
        flags.append("low_confidence_extraction")

    return ExtractionResult(
        fields=fields,
        fields_flat=flat,
        backend=backend,
        document_type=document_type,
        confidence=confidence,
        flags=flags,
        page_index=page_index,
    )
