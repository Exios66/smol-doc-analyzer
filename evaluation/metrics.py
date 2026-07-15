"""
metrics.py

Scores raw eval_results.jsonl records per task and produces a per-(task,
backend) summary table -- the direct input to the cost model spreadsheet's
"Eval Results" sheet.

Classification: exact-match accuracy + macro F1 across taxonomy labels.
Extraction: field-level precision/recall/F1 against the synthetic ground
    truth skeleton (exact match by default; swap in a fuzzy/normalized
    comparator for free-text fields like addresses).
Memo generation: no single ground-truth string exists, so score via:
    (a) rubric coverage -- did the memo include the required fields
        (claim number, coverage determination, next steps, etc.)
    (b) LLM-judge score -- a frontier model rates 1-5 on accuracy/
        completeness/tone against the source documents (use the
        *same* judge model across all three backends' outputs to keep
        scoring fair, and log judge cost separately from generation cost).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskBackendSummary:
    task: str
    backend: str
    model_id: str
    n_examples: int
    accuracy: float | None  # classification
    macro_f1: float | None  # classification / extraction
    judge_score_avg: float | None  # memo_generation (1-5 scale)
    rubric_coverage: float | None  # memo_generation (0-1, required fields present)
    avg_latency_seconds: float
    total_cost_usd: float
    avg_cost_per_doc_usd: float
    error_rate: float


DEFAULT_FUZZY_FIELDS = frozenset(
    {
        "location",
        "loss_location",
        "policyholder_name",
        "insured",
        "adjuster_assigned",
        "description",
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


def score_classification(records: list[dict]) -> tuple[float, float]:
    normalized = [
        {
            **r,
            "prediction": normalize_label(r.get("prediction")),
            "ground_truth": normalize_label(r.get("ground_truth")),
        }
        for r in records
    ]
    correct = sum(1 for r in normalized if r["prediction"] == r["ground_truth"])
    accuracy = correct / len(normalized) if normalized else 0.0

    labels = sorted({r["ground_truth"] for r in normalized if r["ground_truth"]})
    f1s = []
    for label in labels:
        tp = sum(
            1
            for r in normalized
            if r["prediction"] == label and r["ground_truth"] == label
        )
        fp = sum(
            1
            for r in normalized
            if r["prediction"] == label and r["ground_truth"] != label
        )
        fn = sum(
            1
            for r in normalized
            if r["prediction"] != label and r["ground_truth"] == label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return accuracy, macro_f1


def score_extraction(
    records: list[dict], fuzzy_fields: set[str] | None = None
) -> float:
    """Field-level micro-F1 across all examples.

    ``fuzzy_fields`` get normalized (lowercase, whitespace-collapsed) comparison
    instead of exact match.

    A wrong predicted value against a present ground-truth value counts as both
    a false positive and a false negative (standard multi-label micro-F1).
    """
    if fuzzy_fields is None:
        fuzzy_fields = set(DEFAULT_FUZZY_FIELDS)
    tp = fp = fn = 0
    for r in records:
        pred = r["prediction"] if isinstance(r["prediction"], dict) else {}
        truth = r["ground_truth"] if isinstance(r["ground_truth"], dict) else {}
        # Ignore harness parse-error markers.
        pred = {k: v for k, v in pred.items() if not str(k).startswith("_")}
        all_fields = set(pred) | set(truth)
        for field in all_fields:
            p_val, t_val = _as_comparable(pred.get(field)), _as_comparable(truth.get(field))
            fuzzy = field in fuzzy_fields
            if p_val is not None and t_val is not None:
                if _values_equal(p_val, t_val, fuzzy=fuzzy):
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif p_val is not None and t_val is None:
                fp += 1
            elif p_val is None and t_val is not None:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )


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
        if task == "classification" and clean:
            accuracy, macro_f1 = score_classification(clean)
        elif task == "extraction" and clean:
            macro_f1 = score_extraction(clean, fuzzy_fields=fuzzy_fields)
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
            )
        )
    return summaries


def write_summary_csv(summaries: list[TaskBackendSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        out_path.write_text("", encoding="utf-8")
        print(f"Wrote empty summary -> {out_path}")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
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
