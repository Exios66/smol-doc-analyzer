"""Evaluate predicted claim outcomes against gold expected_outcome labels.

Adds outcome accuracy / macro-F1 / confusion to the evaluation battery alongside
classification and extraction reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.pipeline.orchestrator import DocumentAnalysisOrchestrator
from src.pipeline.outcome import OUTCOME_LABELS, derive_expected_outcome, features_from_skeleton
from src.pipeline.types import AnalysisDocument
from src.utils.config import Config
from src.utils.io import load_jsonl, write_json
from src.utils.wandb_utils import load_wandb_settings, start_run

logger = logging.getLogger(__name__)


def _gold_outcome(row: dict[str, Any]) -> str | None:
    if isinstance(row.get("expected_outcome"), str) and row["expected_outcome"]:
        return row["expected_outcome"]
    skeleton = row.get("skeleton")
    if isinstance(skeleton, dict):
        if isinstance(skeleton.get("expected_outcome"), str) and skeleton["expected_outcome"]:
            return skeleton["expected_outcome"]
        return derive_expected_outcome(features_from_skeleton(skeleton))
    return None


def evaluate_outcomes(
    rows: list[dict[str, Any]],
    *,
    cfg: Config | None = None,
    enable_vision: bool | None = False,
    classifier_dir: Path | None = None,
    extractor_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the analysis chain and score predicted vs gold expected_outcome."""
    cfg = cfg or Config.load()
    orch = DocumentAnalysisOrchestrator(
        cfg=cfg,
        enable_vision=enable_vision,
        classifier_dir=classifier_dir,
        extractor_dir=extractor_dir,
    )

    y_true: list[str] = []
    y_pred: list[str] = []
    details: list[dict[str, Any]] = []
    skipped = 0

    for row in rows:
        gold = _gold_outcome(row)
        if not gold:
            skipped += 1
            continue

        # Ensure gold is visible to PredictOutcomeStage via metadata.
        enriched = dict(row)
        meta = dict(enriched.get("metadata") or {})
        if "skeleton" in enriched and "skeleton" not in meta:
            meta["skeleton"] = enriched["skeleton"]
        meta.setdefault("expected_outcome", gold)
        if isinstance(enriched.get("skeleton"), dict):
            sk = dict(enriched["skeleton"])
            sk.setdefault("expected_outcome", gold)
            meta["skeleton"] = sk
        enriched["metadata"] = meta
        enriched["expected_outcome"] = gold

        ctx = orch.analyze(AnalysisDocument.from_row(enriched))
        payload = ctx.to_dict()
        pred = (payload.get("outcome") or {}).get("expected_outcome") or "investigate"
        y_true.append(gold)
        y_pred.append(pred)
        details.append(
            {
                "record_id": payload.get("record_id"),
                "claim_id": payload.get("claim_id"),
                "gold": gold,
                "pred": pred,
                "correct": gold == pred,
                "confidence": (payload.get("outcome") or {}).get("confidence"),
                "features": (payload.get("outcome") or {}).get("features"),
            }
        )

    labels = list(OUTCOME_LABELS)
    if not y_true:
        report = {
            "n": 0,
            "n_skipped_no_gold": skipped,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": [],
            "label_order": labels,
            "details": [],
        }
        return report

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_class = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    gold_dist = dict(Counter(y_true))
    pred_dist = dict(Counter(y_pred))

    report = {
        "n": len(y_true),
        "n_skipped_no_gold": skipped,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "label_order": labels,
        "gold_distribution": gold_dist,
        "pred_distribution": pred_dist,
        "n_correct": sum(1 for d in details if d["correct"]),
        "details": details,
        "chain": orch.stage_names,
        "metric_family": "claim_outcome_prediction",
    }
    return report


def write_outcome_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "outcome_prediction_report.json"
    md_path = out_dir / "outcome_prediction_report.md"
    # Keep JSON lean for dashboards (details can be large)
    slim = {k: v for k, v in report.items() if k != "details"}
    slim["n_detail_rows"] = len(report.get("details") or [])
    write_json(json_path, slim)

    lines = [
        "# Claim outcome prediction report",
        "",
        f"- N scored: **{report.get('n')}**",
        f"- Skipped (no gold): {report.get('n_skipped_no_gold')}",
        f"- Accuracy: **{report.get('accuracy')}**",
        f"- Macro F1: **{report.get('macro_f1')}**",
        f"- Chain: `{' → '.join(report.get('chain') or [])}`",
        "",
        "## Label distribution (gold)",
        "",
    ]
    for lab, n in sorted((report.get("gold_distribution") or {}).items()):
        lines.append(f"- `{lab}`: {n}")
    lines.extend(["", "## Per-class F1", ""])
    per = report.get("per_class") or {}
    for lab in report.get("label_order") or OUTCOME_LABELS:
        stats = per.get(lab) or {}
        if not stats:
            continue
        lines.append(
            f"- `{lab}`: precision={stats.get('precision', 0):.3f} "
            f"recall={stats.get('recall', 0):.3f} f1={stats.get('f1-score', 0):.3f} "
            f"support={stats.get('support', 0)}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Gold `expected_outcome` is a deterministic function of skeleton features "
            "(complexity, injuries, damage vs deductible/reserve). Accuracy therefore "
            "tracks how well upstream extraction recovers those features for the "
            "predictive disposition rule — complementary to classification accuracy "
            "and extraction field F1.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate claim outcome prediction accuracy vs gold expected_outcome"
    )
    parser.add_argument(
        "--in",
        dest="inp",
        type=Path,
        required=True,
        help="JSONL of documents with skeleton and/or expected_outcome gold labels",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Report directory (default: evaluation/reports)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--no-vision", action="store_true", default=True)
    parser.add_argument("--classifier-dir", type=Path, default=None)
    parser.add_argument("--extractor-dir", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = Config.load()
    rows = load_jsonl(args.inp)
    if args.limit is not None:
        rows = rows[: args.limit]

    enable_vision = True if args.vision else False
    report = evaluate_outcomes(
        rows,
        cfg=cfg,
        enable_vision=enable_vision,
        classifier_dir=args.classifier_dir,
        extractor_dir=args.extractor_dir,
    )
    out_dir = args.out_dir or cfg.evaluation_reports_dir
    json_path, md_path = write_outcome_report(report, out_dir)

    use_wandb = False if args.no_wandb else (True if args.wandb else None)
    settings = load_wandb_settings(enabled=False if args.no_wandb else use_wandb)
    with start_run(
        name="outcome-prediction-eval",
        job_type="eval",
        config={"n": report.get("n"), "input": str(args.inp)},
        tags=["outcome", "eval"],
        settings=settings,
    ) as wb:
        if report.get("accuracy") is not None:
            wb.summary(
                {
                    "accuracy": report["accuracy"],
                    "macro_f1": report["macro_f1"],
                    "n": report["n"],
                }
            )
            wb.log(
                {
                    "eval/outcome_accuracy": report["accuracy"],
                    "eval/outcome_macro_f1": report["macro_f1"],
                    "eval/n": report["n"],
                }
            )
            wb.log_artifact_files(
                name="outcome-prediction-report",
                paths=[json_path, md_path],
                artifact_type="evaluation",
                metadata={
                    "accuracy": report["accuracy"],
                    "macro_f1": report["macro_f1"],
                },
            )

    print(
        json.dumps(
            {
                "accuracy": report.get("accuracy"),
                "macro_f1": report.get("macro_f1"),
                "n": report.get("n"),
            },
            indent=2,
        )
    )
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
