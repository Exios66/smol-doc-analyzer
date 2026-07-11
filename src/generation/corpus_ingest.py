"""
Ingest small public corpus samples used as prime examples for characteristic profiling.

Downloads are capped and cached under data/raw/. Bulk binaries are gitignored;
committed profiles under data/profiles/ remain the reproducible source of truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.config import Config
from src.utils.io import append_jsonl, write_json
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)

LEGAL_SAMPLE_TEXTS = [
    (
        "As a threshold matter, the duty to defend is broader than the duty to indemnify. "
        "On the present record, timely notice of loss was provided, and the facts as presently "
        "developed indicate that coverage appears implicated under the named peril. "
        "Notwithstanding the claimant's alternative theory, proximate cause is most consistent "
        "with the documented water intrusion path."
    ),
    (
        "Issue: whether the exclusion bars recovery. Rule: exclusions are construed narrowly. "
        "Application: the material fact in dispute is the sequence of events preceding the loss. "
        "Conclusion: reservation of rights is appropriate pending further investigation."
    ),
    (
        "The insured owed a duty of reasonable care to maintain the premises. Breach is alleged "
        "in the failure to warn of a foreseeable risk. Causation and damages remain contested. "
        "Accordingly, comparative negligence may allocate responsibility on these facts."
    ),
    (
        "For the reasons set forth below, the adjuster's coverage determination should address "
        "whether additional insured status attaches and whether subrogation interest is preserved. "
        "In light of the foregoing, proof of loss and an independent medical examination may be warranted."
    ),
    (
        "By contrast, the statement under oath is inconsistent with the contemporaneous incident report. "
        "It follows that mitigation of damages and bad faith claim handling risk should be monitored "
        "while the factual dispute regarding notice is resolved."
    ),
]


def _safe_import_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required for corpus ingest") from exc
    return load_dataset


def ingest_funsd(raw_dir: Path, n: int = 50) -> list[dict[str, Any]]:
    load_dataset = _safe_import_datasets()
    out: list[dict[str, Any]] = []
    try:
        ds = load_dataset("nielsr/funsd", split="train")
    except Exception as exc:
        logger.warning("FUNSD download failed (%s); writing empty FUNSD sample stub", exc)
        return out

    field_labels: list[str] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        words = row.get("words") or row.get("tokens") or []
        ner = row.get("ner_tags") or []
        # Collect likely question/header tokens as field-label candidates
        for w, tag in zip(words, ner):
            if isinstance(tag, int) and tag in (1, 2) and isinstance(w, str) and len(w) > 2:
                field_labels.append(w)
        sample = {
            "source": "nielsr/funsd",
            "sample_id": f"funsd-{i}",
            "n_words": len(words),
            "field_label_candidates": field_labels[-20:],
            "license_note": "FUNSD research dataset; used for field lexicon / layout shape only",
        }
        out.append(sample)
        field_labels = field_labels[-200:]
    write_json(raw_dir / "funsd_samples.json", {"samples": out, "field_labels": sorted(set(field_labels))[:200]})
    return out


def ingest_doclaynet(raw_dir: Path, n: int = 30) -> list[dict[str, Any]]:
    load_dataset = _safe_import_datasets()
    out: list[dict[str, Any]] = []
    try:
        ds = load_dataset("ds4sd/DocLayNet", split="train", streaming=True)
    except Exception as exc:
        logger.warning("DocLayNet download failed (%s); skipping", exc)
        return out

    layout_counts: Counter[str] = Counter()
    legalish_text: list[str] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        cats = row.get("category_id") or []
        for c in cats:
            layout_counts[str(c)] += 1
        meta = row.get("metadata") or {}
        doc_category = str(meta.get("doc_category") or meta.get("collection") or "")
        cells = row.get("pdf_cells") or []
        text_bits: list[str] = []
        for cell_group in cells[:5]:
            if isinstance(cell_group, list):
                for cell in cell_group[:10]:
                    if isinstance(cell, dict) and cell.get("text"):
                        text_bits.append(str(cell["text"]))
        joined = " ".join(text_bits)
        if "law" in doc_category.lower() or "regulat" in doc_category.lower():
            if joined:
                legalish_text.append(joined[:500])
        out.append(
            {
                "source": "ds4sd/DocLayNet",
                "sample_id": f"doclaynet-{i}",
                "doc_category": doc_category,
                "n_boxes": len(cats),
                "license_note": "DocLayNet; layout class mix and prose texture only",
            }
        )
    write_json(
        raw_dir / "doclaynet_samples.json",
        {
            "samples": out,
            "layout_id_counts": dict(layout_counts),
            "legalish_text_snippets": legalish_text[:20],
        },
    )
    return out


def ingest_rvl_cdip(raw_dir: Path, n_per_class: int = 5) -> list[dict[str, Any]]:
    load_dataset = _safe_import_datasets()
    wanted = {"form", "letter", "email", "memo", "invoice", "questionnaire", "specification", "budget"}
    label_names = {
        0: "letter",
        1: "form",
        2: "email",
        3: "handwritten",
        4: "advertisement",
        5: "scientific report",
        6: "scientific publication",
        7: "specification",
        8: "file folder",
        9: "news article",
        10: "budget",
        11: "invoice",
        12: "presentation",
        13: "questionnaire",
        14: "resume",
        15: "memo",
    }
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    try:
        ds = load_dataset("aharley/rvl_cdip", split="train", streaming=True)
    except Exception as exc:
        logger.warning("RVL-CDIP download failed (%s); writing style stub only", exc)
        write_json(raw_dir / "rvl_cdip_samples.json", {"samples": [], "note": str(exc)})
        return out

    for i, row in enumerate(ds):
        if len(out) >= n_per_class * len(wanted):
            break
        if i > 5000:
            break
        label_id = int(row.get("label", -1))
        name = label_names.get(label_id, "unknown")
        if name not in wanted or counts[name] >= n_per_class:
            continue
        counts[name] += 1
        out.append(
            {
                "source": "aharley/rvl_cdip",
                "sample_id": f"rvl-{name}-{counts[name]}",
                "rvl_label": name,
                "license_note": "RVL-CDIP subset of IIT-CDIP; surface style only",
            }
        )
    write_json(raw_dir / "rvl_cdip_samples.json", {"samples": out, "counts": dict(counts)})
    return out


def ingest_legal_writing(raw_dir: Path) -> list[dict[str, Any]]:
    """Bundle curated public-style legal writing samples (vocab/reasoning only)."""
    samples = []
    for i, text in enumerate(LEGAL_SAMPLE_TEXTS):
        samples.append(
            {
                "source": "bundled_legal_style_samples",
                "sample_id": f"legal-style-{i}",
                "text": text,
                "license_note": "Synthetic exemplars of public legal writing style; not case filings",
            }
        )
    # Optionally enrich from Hub if available
    try:
        load_dataset = _safe_import_datasets()
        ds = load_dataset("pile-of-law/pile-of-law", "r_legaldvice", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= 10:
                break
            text = str(row.get("text") or "")[:800]
            if len(text) < 80:
                continue
            samples.append(
                {
                    "source": "pile-of-law/pile-of-law:r_legaldvice",
                    "sample_id": f"pol-{i}",
                    "text": text,
                    "license_note": "pile-of-law sample; vocabulary/style only, not taxonomy labels",
                }
            )
    except Exception as exc:
        logger.info("Optional pile-of-law enrich skipped: %s", exc)

    write_json(raw_dir / "legal_writing_samples.json", {"samples": samples})
    return samples


def write_manifest(raw_dir: Path, entries: list[dict[str, Any]]) -> Path:
    path = raw_dir / "manifest.jsonl"
    if path.exists():
        path.unlink()
    append_jsonl(path, entries)
    return path


def run_ingest(cfg: Config, funsd_n: int = 50, doclaynet_n: int = 30, rvl_n: int = 5) -> dict[str, int]:
    raw_dir = cfg.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []

    funsd = ingest_funsd(raw_dir, n=funsd_n)
    counts["funsd"] = len(funsd)
    manifest.extend(funsd)

    doclay = ingest_doclaynet(raw_dir, n=doclaynet_n)
    counts["doclaynet"] = len(doclay)
    manifest.extend(doclay)

    rvl = ingest_rvl_cdip(raw_dir, n_per_class=rvl_n)
    counts["rvl_cdip"] = len(rvl)
    manifest.extend(rvl)

    legal = ingest_legal_writing(raw_dir)
    counts["legal_writing"] = len(legal)
    manifest.extend(
        {
            "source": s["source"],
            "sample_id": s["sample_id"],
            "local_path": str(raw_dir / "legal_writing_samples.json"),
            "license_note": s["license_note"],
        }
        for s in legal
    )

    write_manifest(raw_dir, manifest)
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id="corpus-ingest-batch",
            stage="corpus_ingest",
            source="public_hub_and_bundled_samples",
            prompt_version="corpus_ingest_v1",
            model=None,
            extra={"counts": counts},
        ),
    )
    write_json(raw_dir / "ingest_summary.json", counts)
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest small public corpus samples")
    parser.add_argument("--funsd-n", type=int, default=50)
    parser.add_argument("--doclaynet-n", type=int, default=30)
    parser.add_argument("--rvl-n", type=int, default=5)
    args = parser.parse_args()
    cfg = Config.load()
    counts = run_ingest(cfg, funsd_n=args.funsd_n, doclaynet_n=args.doclaynet_n, rvl_n=args.rvl_n)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
