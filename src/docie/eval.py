"""
DICIE evaluation — paper Table I (classification) + Table II (extraction).

Classification metrics: accuracy, AUC (OVR / OVO), confusion matrix.
Extraction metrics: per-field precision, recall, and F1 = 2PR/(P+R).

Usage:
  python -m src.docie.eval --application salvage_claims
  python -m src.docie.eval --application medical_bills
  python -m src.docie.eval --all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from evaluation.metrics import classification_metrics, extraction_metrics
from src.docie.applications import list_applications, load_application
from src.docie.pipeline import DociePipeline
from src.utils.config import Config, REPO_ROOT
from src.utils.io import load_jsonl, write_json

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET = REPO_ROOT / "data" / "eval" / "docie_eval_set.jsonl"
PAPER_APPS = ("medical_bills", "salvage_claims")

# Paper Table II field order (taxonomy keys).
PAPER_FIELDS: dict[str, list[str]] = {
    "salvage_claims": ["claim_id", "vin", "make", "model", "year"],
    "medical_bills": ["name", "dob", "claim_id", "patient_id", "address"],
}

DICIE_FUZZY_FIELDS = frozenset({"name", "address"})


def _flat_fields(prediction_fields: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (prediction_fields or {}).items():
        if v is None:
            continue
        if isinstance(v, list):
            out[k] = v[0] if v else None
        else:
            out[k] = v
    return out


def _normalize_truth(fields: dict[str, Any] | None) -> dict[str, Any]:
    """Drop null gold values so absent fields are not false negatives."""
    out: dict[str, Any] = {}
    for k, v in (fields or {}).items():
        if v is None or v == "":
            continue
        out[k] = v
    return out


def evaluate_application(
    application: str,
    *,
    eval_path: Path,
    cfg: Config | None = None,
    run_ocr: bool = False,
    vit_model_dir: Path | None = None,
    extractor_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run DICIE on gold rows and return paper-style metric payload."""
    cfg = cfg or Config.load()
    profile = load_application(application)
    rows = [
        r
        for r in load_jsonl(eval_path)
        if str(r.get("application", "")).lower().replace("-", "_") == profile.name
    ]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No eval rows for application={application!r} in {eval_path}")

    pipe = DociePipeline(
        application=profile.name,
        cfg=cfg,
        run_ocr=run_ocr,
        vit_model_dir=vit_model_dir,
        extractor_dir=extractor_dir,
    )

    y_true: list[str] = []
    y_pred: list[str] = []
    y_scores: list[dict[str, float]] = []
    ext_records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for row in rows:
        record_id = str(row.get("record_id") or "unknown")
        gold_label = str(row.get("document_type") or "")
        gold_fields = _normalize_truth(row.get("ground_truth_fields") or {})
        prediction = pipe.process(
            record_id=record_id,
            text=row.get("text"),
            pdf_path=row.get("pdf_path"),
            image_path=row.get("image_path"),
            source_path=row.get("source_path") or row.get("path"),
            metadata={"eval": True},
        )
        pred_label = prediction.classification.label
        # Prefer aggregated page score maps; fall back to doc-level one-hot-ish confidence.
        page_scores = prediction.classification.page_predictions[0].scores if prediction.classification.page_predictions else {}
        if not page_scores:
            page_scores = {pred_label: prediction.classification.confidence}

        y_true.append(gold_label)
        y_pred.append(pred_label)
        y_scores.append({lab: float(page_scores.get(lab, 0.0)) for lab in profile.labels})

        pred_fields = _flat_fields(prediction.extraction.fields_flat)
        ext_records.append({"prediction": pred_fields, "ground_truth": gold_fields})
        details.append(
            {
                "record_id": record_id,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "correct": pred_label == gold_label,
                "classification_confidence": prediction.classification.confidence,
                "gold_fields": gold_fields,
                "pred_fields": pred_fields,
                "classification_backend": prediction.classification.backend,
                "extraction_backend": prediction.extraction.backend,
            }
        )

    field_list = PAPER_FIELDS.get(profile.name, list(profile.extraction_fields))
    clf = classification_metrics(
        y_true,
        y_pred,
        y_scores=y_scores,
        labels=profile.labels,
    )
    ext = extraction_metrics(
        ext_records,
        fuzzy_fields=set(DICIE_FUZZY_FIELDS),
        fields=field_list,
    )

    return {
        "application": profile.name,
        "n": len(rows),
        "eval_path": str(eval_path),
        "chain": [
            "document_processing",
            "document_classification",
            "information_extraction",
            "output_aggregation",
        ],
        "classification": clf,
        "extraction": ext,
        "details": details,
        "paper_reference": {
            "classification_table": "Table I (Accuracy, AUC OVR/OVO)",
            "extraction_table": "Table II (per-field Precision / Recall / F1)",
            "f1_definition": "2 * precision * recall / (precision + recall)",
        },
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render paper-style markdown tables."""
    app = payload["application"]
    clf = payload["classification"]
    ext = payload["extraction"]
    lines = [
        f"# DICIE evaluation — `{app}`",
        "",
        "Paper-aligned metrics from Raj, Dickinson & Fung "
        "(*Document Classification and Information Extraction framework for Insurance Applications*).",
        "",
        f"- N: **{payload['n']}**",
        f"- Eval set: `{payload['eval_path']}`",
        "",
        "## Table I — Document Classification",
        "",
        f"| Metric | Score |",
        f"|--------|------:|",
        f"| Accuracy | {clf['accuracy']:.4f} |",
        f"| AUC (OVR) | {_fmt(clf.get('auc_ovr'))} |",
        f"| AUC (OVO) | {_fmt(clf.get('auc_ovo'))} |",
        f"| Macro F1 (diagnostic) | {clf['macro_f1']:.4f} |",
        f"| Micro F1 (diagnostic) | {clf['micro_f1']:.4f} |",
        f"| Weighted F1 (diagnostic) | {clf['weighted_f1']:.4f} |",
        "",
        "### Per-class F1",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for label in clf.get("labels") or []:
        stats = (clf.get("per_class") or {}).get(label) or {}
        lines.append(
            f"| {label} | {stats.get('precision', 0):.4f} | {stats.get('recall', 0):.4f} | "
            f"{stats.get('f1', 0):.4f} | {int(stats.get('support', 0))} |"
        )

    lines.extend(
        [
            "",
            "### Confusion matrix",
            "",
            f"Labels order: `{clf.get('labels')}`",
            "",
            "```",
            json.dumps(clf.get("confusion_matrix"), indent=2),
            "```",
            "",
            "## Table II — Information Extraction",
            "",
            f"| Aggregate | Score |",
            f"|-----------|------:|",
            f"| Micro Precision | {ext['micro_precision']:.4f} |",
            f"| Micro Recall | {ext['micro_recall']:.4f} |",
            f"| Micro F1 | {ext['micro_f1']:.4f} |",
            f"| Macro F1 (field mean) | {ext['macro_f1']:.4f} |",
            "",
            "### Per-field Precision / Recall / F1",
            "",
            "| Field | Precision | Recall | F1 | Support |",
            "|-------|----------:|-------:|---:|--------:|",
        ]
    )
    for field_name in ext.get("fields") or []:
        stats = (ext.get("per_field") or {}).get(field_name) or {}
        lines.append(
            f"| {field_name} | {stats.get('precision', 0):.4f} | {stats.get('recall', 0):.4f} | "
            f"{stats.get('f1', 0):.4f} | {int(stats.get('support', 0))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}"


def write_reports(
    payload: dict[str, Any],
    *,
    reports_dir: Path,
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    app = payload["application"]
    json_path = reports_dir / f"docie_{app}_metrics.json"
    md_path = reports_dir / f"docie_{app}_report.md"
    # Persist without the bulky per-row dump duplicated — keep details in JSON.
    write_json(json_path, payload)
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate DICIE with paper Table I / Table II metrics"
    )
    parser.add_argument(
        "--application",
        "-a",
        choices=list_applications(),
        default=None,
        help="Application profile to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate medical_bills and salvage_claims",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=DEFAULT_EVAL_SET,
        help="Gold JSONL with document_type + ground_truth_fields",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ocr", action="store_true", help="Enable OCR during Stage 1")
    parser.add_argument("--vit-model-dir", type=Path, default=None)
    parser.add_argument("--extractor-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    apps: list[str]
    if args.all:
        apps = list(PAPER_APPS)
    elif args.application:
        apps = [args.application]
    else:
        parser.error("Provide --application or --all")

    cfg = Config.load()
    out_dir = args.out_dir or cfg.evaluation_reports_dir

    for app in apps:
        payload = evaluate_application(
            app,
            eval_path=args.eval_set,
            cfg=cfg,
            run_ocr=args.ocr,
            vit_model_dir=args.vit_model_dir,
            extractor_dir=args.extractor_dir,
            limit=args.limit,
        )
        json_path, md_path = write_reports(payload, reports_dir=out_dir)
        clf = payload["classification"]
        ext = payload["extraction"]
        print(
            f"[{app}] n={payload['n']}  "
            f"acc={clf['accuracy']:.4f}  "
            f"auc_ovr={_fmt(clf.get('auc_ovr'))}  "
            f"auc_ovo={_fmt(clf.get('auc_ovo'))}  "
            f"ext_micro_f1={ext['micro_f1']:.4f}"
        )
        print(f"  -> {json_path}")
        print(f"  -> {md_path}")


if __name__ == "__main__":
    main()
