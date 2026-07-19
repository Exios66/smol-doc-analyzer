"""
metrics.py

Scores raw eval_results.jsonl records per task and produces a per-(task,
backend) summary table -- the direct input to the cost model spreadsheet's
"Eval Results" sheet.

Paper-aligned (Raj, Dickinson & Fung — DICIE):
  Classification: accuracy + AUC (OVR / OVO) + confusion matrix (Table I).
  Extraction: per-field precision / recall / F1 = 2PR/(P+R) (Table II).

Also retained for the cost harness / ACORD path:
  Classification: macro / micro / weighted F1.
  Extraction: field-level micro-F1 aggregates.
  Memo generation: rubric coverage + optional LLM-judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class TaskBackendSummary:
    task: str
    backend: str
    model_id: str
    n_examples: int
    accuracy: float | None  # classification
    macro_f1: float | None  # classification macro F1 / extraction micro-F1 (legacy key)
    judge_score_avg: float | None  # memo_generation (1-5 scale)
    rubric_coverage: float | None  # memo_generation (0-1, required fields present)
    avg_latency_seconds: float
    total_cost_usd: float
    avg_cost_per_doc_usd: float
    error_rate: float
    # Paper / extended metrics (optional; omitted from older CSV consumers via default)
    micro_f1: float | None = None
    weighted_f1: float | None = None
    auc_ovr: float | None = None
    auc_ovo: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


DEFAULT_FUZZY_FIELDS = frozenset(
    {
        "location",
        "loss_location",
        "policyholder_name",
        "insured",
        "adjuster_assigned",
        "description",
        "name",
        "address",
    }
)

REQUIRED_MEMO_FIELDS = [
    "claim_number",
    "coverage_determination",
    "next_steps",
    "adjuster_notes",
]

# Keyword aliases so rubric checks match both the abstract field names and the
# phrasing used by the project's template / frontier memo prompts.
_MEMO_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "claim_number": ("claim number", "claim #", "claim id", "acord form", "clm-"),
    "coverage_determination": (
        "coverage determination",
        "coverage grant",
        "whether coverage",
        "coverage decision",
    ),
    "next_steps": ("next steps", "next step"),
    "adjuster_notes": ("adjuster memo", "adjuster notes", "adjuster analysis"),
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_label(value: Any) -> str:
    """Normalize taxonomy labels (spaces/hyphens → underscores, strip punctuation)."""
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _as_comparable(value: Any) -> Any:
    """Flatten list-valued extractor outputs to a single comparable scalar."""
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            return value[0]
        return " | ".join(str(v) for v in value)
    return value


def _values_equal(p_val: Any, t_val: Any, *, fuzzy: bool) -> bool:
    if p_val is None or t_val is None:
        return False
    if fuzzy:
        return _normalize_text(str(p_val)) == _normalize_text(str(t_val))
    if isinstance(p_val, str) and isinstance(t_val, str):
        return p_val == t_val
    # Coerce numeric / mixed types via string form for exact equality.
    return str(p_val).strip() == str(t_val).strip()


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (paper Table II F1)."""
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _normalize_score_row(
    scores: dict[str, float] | None,
    labels: Sequence[str],
) -> list[float]:
    """Convert a label→score map into a simplex over ``labels``."""
    raw = [float((scores or {}).get(lab, 0.0)) for lab in labels]
    total = sum(max(0.0, v) for v in raw)
    if total <= 0:
        # Uniform when no scores (avoids AUC crash on empty rows).
        n = len(labels) or 1
        return [1.0 / n] * len(labels)
    return [max(0.0, v) / total for v in raw]


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    y_scores: Sequence[dict[str, float]] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Paper Table I metrics (+ secondary F1 diagnostics).

    Returns accuracy, auc_ovr, auc_ovo, confusion_matrix, and macro/micro/weighted F1.
    AUC requires per-example score dicts (``y_scores``); otherwise AUC fields are None.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    true_n = [normalize_label(v) for v in y_true]
    pred_n = [normalize_label(v) for v in y_pred]
    if labels is None:
        labels = sorted({*true_n, *pred_n} - {""})
    else:
        labels = [normalize_label(x) for x in labels]

    if not true_n:
        return {
            "accuracy": 0.0,
            "auc_ovr": None,
            "auc_ovo": None,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "weighted_f1": 0.0,
            "labels": list(labels),
            "confusion_matrix": [],
            "per_class": {},
            "n": 0,
        }

    acc = float(accuracy_score(true_n, pred_n))
    cm = confusion_matrix(true_n, pred_n, labels=list(labels)).tolist()
    macro = float(f1_score(true_n, pred_n, labels=list(labels), average="macro", zero_division=0))
    micro = float(f1_score(true_n, pred_n, labels=list(labels), average="micro", zero_division=0))
    weighted = float(
        f1_score(true_n, pred_n, labels=list(labels), average="weighted", zero_division=0)
    )

    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true_n, pred_n) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_n, pred_n) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_n, pred_n) if t == label and p != label)
        support = sum(1 for t in true_n if t == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": float(support),
        }

    auc_ovr = auc_ovo = None
    if y_scores is not None and len(y_scores) == len(true_n) and len(labels) >= 2:
        try:
            # sklearn requires multiclass ``labels`` to be lexicographically sorted.
            auc_labels = sorted(labels)
            proba = np.asarray(
                [_normalize_score_row(s, auc_labels) for s in y_scores], dtype=float
            )
            present = {t for t in true_n}
            if len(present) >= 2:
                auc_ovr = float(
                    roc_auc_score(
                        true_n, proba, multi_class="ovr", labels=auc_labels
                    )
                )
                auc_ovo = float(
                    roc_auc_score(
                        true_n, proba, multi_class="ovo", labels=auc_labels
                    )
                )
        except ValueError:
            auc_ovr = auc_ovo = None

    return {
        "accuracy": acc,
        "auc_ovr": auc_ovr,
        "auc_ovo": auc_ovo,
        "macro_f1": macro,
        "micro_f1": micro,
        "weighted_f1": weighted,
        "labels": list(labels),
        "confusion_matrix": cm,
        "per_class": per_class,
        "n": len(true_n),
    }


def extraction_metrics(
    records: list[dict],
    fuzzy_fields: set[str] | None = None,
    *,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Paper Table II metrics: per-field precision / recall / F1 (harmonic mean).

    Also returns micro- and macro-averaged aggregates over fields.
    A wrong predicted value against a present ground-truth value counts as both
    a false positive and a false negative (standard multi-label micro-F1).
    """
    if fuzzy_fields is None:
        fuzzy_fields = set(DEFAULT_FUZZY_FIELDS)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    seen_fields: list[str] = []

    for r in records:
        pred = r["prediction"] if isinstance(r.get("prediction"), dict) else {}
        truth = r["ground_truth"] if isinstance(r.get("ground_truth"), dict) else {}
        pred = {k: v for k, v in pred.items() if not str(k).startswith("_")}
        if fields is not None:
            all_fields = list(fields)
        else:
            all_fields = sorted(set(pred) | set(truth))
            for fname in all_fields:
                if fname not in seen_fields:
                    seen_fields.append(fname)
        for fname in all_fields:
            p_val = _as_comparable(pred.get(fname))
            t_val = _as_comparable(truth.get(fname))
            fuzzy = fname in fuzzy_fields
            if p_val is not None and t_val is not None:
                if _values_equal(p_val, t_val, fuzzy=fuzzy):
                    counts[fname]["tp"] += 1
                else:
                    counts[fname]["fp"] += 1
                    counts[fname]["fn"] += 1
            elif p_val is not None and t_val is None:
                counts[fname]["fp"] += 1
            elif p_val is None and t_val is not None:
                counts[fname]["fn"] += 1

    field_order = list(fields) if fields is not None else seen_fields

    per_field: dict[str, dict[str, float]] = {}
    tp = fp = fn = 0
    f1s: list[float] = []
    for fname in field_order:
        c = counts[fname]
        tp += c["tp"]
        fp += c["fp"]
        fn += c["fn"]
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = _f1(precision, recall)
        support = float(c["tp"] + c["fn"])
        per_field[fname] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": float(c["tp"]),
            "fp": float(c["fp"]),
            "fn": float(c["fn"]),
        }
        if support > 0:
            f1s.append(f1)

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": _f1(micro_p, micro_r),
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
        "per_field": per_field,
        "fields": list(field_order),
        "n_examples": len(records),
    }


def score_classification(records: list[dict]) -> tuple[float, float]:
    """Backward-compatible: (accuracy, macro_f1). """
    y_true = [r.get("ground_truth") for r in records]
    y_pred = [r.get("prediction") for r in records]
    metrics = classification_metrics(y_true, y_pred)
    return metrics["accuracy"], metrics["macro_f1"]


def score_extraction(
    records: list[dict], fuzzy_fields: set[str] | None = None
) -> float:
    """Backward-compatible: field-level micro-F1 across all examples. """
    return float(extraction_metrics(records, fuzzy_fields=fuzzy_fields)["micro_f1"])


def score_extraction_example(
    prediction: Any, ground_truth: Any, fuzzy_fields: set[str] | None = None
) -> float:
    """Per-example field F1 (used to fill ``EvalResult.score``)."""
    return score_extraction(
        [{"prediction": prediction, "ground_truth": ground_truth}],
        fuzzy_fields=fuzzy_fields,
    )


def score_memo_rubric(
    memo_text: str, required_fields: list[str] | None = None
) -> float:
    """Cheap heuristic: fraction of required concepts mentioned (keyword proxy).

    Replace with a structured check once memo templates stabilize.
    """
    required_fields = required_fields or list(REQUIRED_MEMO_FIELDS)
    if not memo_text:
        return 0.0
    lower = memo_text.lower()
    hits = 0
    for field in required_fields:
        aliases = _MEMO_FIELD_ALIASES.get(field, (field.replace("_", " "),))
        if any(alias in lower for alias in aliases):
            hits += 1
    return hits / len(required_fields)


def annotate_records(
    records: list[dict], fuzzy_fields: set[str] | None = None
) -> list[dict]:
    """Fill per-row ``correct`` / ``score`` fields in-place and return the list."""
    for r in records:
        pred = r.get("prediction")
        parse_error = isinstance(pred, dict) and pred.get("_parse_error") is True
        if r.get("error") or parse_error:
            if parse_error and not r.get("error"):
                r["error"] = "prediction_parse_error"
            r["correct"] = False
            r["score"] = 0.0
            continue
        task = r.get("task")
        truth = r.get("ground_truth")
        if task == "classification":
            ok = normalize_label(pred) == normalize_label(truth)
            r["correct"] = ok
            r["score"] = 1.0 if ok else 0.0
        elif task == "extraction":
            score = score_extraction_example(pred, truth, fuzzy_fields=fuzzy_fields)
            r["score"] = score
            r["correct"] = score >= 1.0
        elif task == "memo_generation":
            score = score_memo_rubric(str(pred or ""))
            r["score"] = score
            r["correct"] = score >= 0.75
            r["score_source"] = "rubric"
            # Prefer LLM-judge score for the row when already present.
            if r.get("judge_score") is not None:
                r["score"] = float(r["judge_score"]) / 5.0
                r["score_source"] = "judge"
                r["correct"] = float(r["judge_score"]) >= 3.75
        else:
            r.setdefault("correct", None)
            r.setdefault("score", None)
    return records


def summarize(
    results_path: Path,
    fuzzy_fields: set[str] | None = None,
    annotate: bool = True,
) -> list[TaskBackendSummary]:
    records = [
        json.loads(line)
        for line in open(results_path, encoding="utf-8")
        if line.strip()
    ]
    if annotate:
        annotate_records(records, fuzzy_fields=fuzzy_fields)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["task"], r["backend"])].append(r)

    summaries: list[TaskBackendSummary] = []
    for (task, backend), group in sorted(grouped.items()):
        errored = [
            r
            for r in group
            if r.get("error")
            or (isinstance(r.get("prediction"), dict) and r["prediction"].get("_parse_error"))
        ]
        clean = [r for r in group if r not in errored]
        n = len(group)

        accuracy = macro_f1 = judge_score = rubric_cov = None
        micro_f1 = weighted_f1 = auc_ovr = auc_ovo = None
        if task == "classification" and clean:
            score_rows = [
                r.get("prediction_scores")
                for r in clean
                if isinstance(r.get("prediction_scores"), dict)
            ]
            cls_m = classification_metrics(
                [r.get("ground_truth") for r in clean],
                [r.get("prediction") for r in clean],
                y_scores=score_rows if len(score_rows) == len(clean) else None,
            )
            accuracy = cls_m["accuracy"]
            macro_f1 = cls_m["macro_f1"]
            micro_f1 = cls_m["micro_f1"]
            weighted_f1 = cls_m["weighted_f1"]
            auc_ovr = cls_m["auc_ovr"]
            auc_ovo = cls_m["auc_ovo"]
        elif task == "extraction" and clean:
            ext_m = extraction_metrics(clean, fuzzy_fields=fuzzy_fields)
            # Legacy summary key ``macro_f1`` stores extraction micro-F1 for the cost sheet.
            macro_f1 = ext_m["micro_f1"]
            micro_f1 = ext_m["micro_f1"]
        elif task == "memo_generation" and clean:
            rubric_cov = sum(score_memo_rubric(str(r["prediction"] or "")) for r in clean) / len(
                clean
            )
            # Prefer an already-merged LLM-judge pass when present; otherwise leave
            # None so judge API calls remain a deliberate, separately costed step.
            judged = [r["judge_score"] for r in clean if r.get("judge_score") is not None]
            if judged:
                judge_score = sum(float(s) for s in judged) / len(judged)

        total_cost = sum(float(r.get("cost_usd") or 0.0) for r in group)
        summaries.append(
            TaskBackendSummary(
                task=task,
                backend=backend,
                model_id=str(group[0].get("model_id") or backend),
                n_examples=n,
                accuracy=accuracy,
                macro_f1=macro_f1,
                judge_score_avg=judge_score,
                rubric_coverage=rubric_cov,
                avg_latency_seconds=(
                    sum(float(r.get("latency_seconds") or 0.0) for r in group) / n if n else 0.0
                ),
                total_cost_usd=total_cost,
                avg_cost_per_doc_usd=total_cost / n if n else 0.0,
                error_rate=len(errored) / n if n else 0.0,
                micro_f1=micro_f1,
                weighted_f1=weighted_f1,
                auc_ovr=auc_ovr,
                auc_ovo=auc_ovo,
            )
        )
    return summaries


_CSV_FIELDS = [
    "task",
    "backend",
    "model_id",
    "n_examples",
    "accuracy",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "auc_ovr",
    "auc_ovo",
    "judge_score_avg",
    "rubric_coverage",
    "avg_latency_seconds",
    "total_cost_usd",
    "avg_cost_per_doc_usd",
    "error_rate",
]


def write_summary_csv(summaries: list[TaskBackendSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        out_path.write_text("", encoding="utf-8")
        print(f"Wrote empty summary -> {out_path}")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    print(f"Wrote summary -> {out_path}")


def write_summary_json(summaries: list[TaskBackendSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in summaries]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote summary -> {out_path}")


def write_annotated_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def score_results_file(
    results_path: Path,
    output_csv: Path,
    output_json: Path | None = None,
    annotated_jsonl: Path | None = None,
    fuzzy_fields: set[str] | None = None,
) -> list[TaskBackendSummary]:
    """Load results, annotate rows, write spreadsheet-ready summary outputs."""
    records = [
        json.loads(line)
        for line in open(results_path, encoding="utf-8")
        if line.strip()
    ]
    annotate_records(records, fuzzy_fields=fuzzy_fields)

    annotated_path = annotated_jsonl or results_path
    write_annotated_jsonl(records, annotated_path)

    summaries = summarize(annotated_path, fuzzy_fields=fuzzy_fields, annotate=False)
    write_summary_csv(summaries, output_csv)
    if output_json is not None:
        write_summary_json(summaries, output_json)
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score eval_results.jsonl into a per-(task, backend) summary table"
    )
    parser.add_argument("--results", type=Path, required=True, help="Path to eval_results.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to write summary.csv")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write summary.json (default: sibling .json of --output)",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        default=None,
        help="Optional path for annotated eval_results (correct/score filled). "
        "Defaults to overwriting --results in place.",
    )
    parser.add_argument(
        "--fuzzy-fields",
        nargs="*",
        default=None,
        help="Extraction fields to compare with normalized/fuzzy matching",
    )
    args = parser.parse_args(argv)

    fuzzy = set(args.fuzzy_fields) if args.fuzzy_fields is not None else None
    output_json = args.output_json
    if output_json is None and args.output.suffix.lower() == ".csv":
        output_json = args.output.with_suffix(".json")

    score_results_file(
        results_path=args.results,
        output_csv=args.output,
        output_json=output_json,
        annotated_jsonl=args.annotated,
        fuzzy_fields=fuzzy,
    )


if __name__ == "__main__":
    main()
