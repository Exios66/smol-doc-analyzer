"""OpenRouter multi-model classify + structured extract PoC for RVL-CDIP samples.

Loads ``recreation_samples.jsonl`` (typically 1040 rows), optionally materializes
images / OCR, queries vision and/or text models via OpenRouter, caches
predictions, and writes classification + extraction-proxy scoreboards.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from evaluation.metrics import normalize_label
from src.rvl_cdip.paths import LABEL_NAMES as RVL_LABELS
from src.rvl_cdip.sample_images import (
    MaterializeResult,
    materialize_samples,
    summarize_materialize,
)
from src.utils.config import REPO_ROOT, Config
from src.utils.llm_client import OpenRouterClient
from src.utils.prompts import load_prompt

DEFAULT_SAMPLES_PATH = (
    REPO_ROOT / "data" / "notebook_demo" / "rvl_cdip_recreation" / "recreation_samples.jsonl"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "notebook_demo" / "rvl_cdip_openrouter_eval"

DEFAULT_VISION_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
)
DEFAULT_TEXT_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.2-3b-instruct:free",
)

EXTRACT_KEYS = (
    "document_type_guess",
    "title_or_subject",
    "parties",
    "dates",
    "identifiers",
    "amounts",
    "addresses_or_locations",
    "key_value_fields",
    "summary",
    "extraction_notes",
)

SYSTEM_PROMPT = (
    "You are a careful document analysis assistant for insurance-style intake. "
    "Follow the task instructions exactly."
)


@dataclass
class PredictionRow:
    document_id: str
    label_id: int
    label: str
    task: str  # classify | extract_annotate
    modality: str  # vision | text
    model_id: str
    prediction: Any
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    dry_run: bool = False
    image_path: str | None = None
    cache_hit: bool = False

    def cache_key(self) -> tuple[str, str, str, str]:
        return (self.document_id, self.model_id, self.task, self.modality)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_samples(
    path: Path | None = None,
    *,
    max_per_class: int | None = None,
    max_total: int | None = None,
) -> list[dict[str, Any]]:
    samples_path = path or DEFAULT_SAMPLES_PATH
    rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_per_class is not None:
        by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_label[int(row["label_id"])].append(row)
        capped: list[dict[str, Any]] = []
        for label_id in range(len(RVL_LABELS)):
            capped.extend(by_label.get(label_id, [])[: max(0, int(max_per_class))])
        rows = capped
    if max_total is not None:
        rows = rows[: max(0, int(max_total))]
    return rows


def label_list_str(labels: Sequence[str] = RVL_LABELS) -> str:
    return "\n".join(f"- {name}" for name in labels)


def normalize_rvl_label(value: Any) -> str:
    """Map model output onto an official RVL label when possible."""
    raw = str(value or "").strip().lower()
    if raw in RVL_LABELS:
        return raw
    norm = normalize_label(raw)
    for name in RVL_LABELS:
        if normalize_label(name) == norm:
            return name
    # Tolerate "scientific_report" style
    spaced = norm.replace("_", " ")
    if spaced in RVL_LABELS:
        return spaced
    return raw


def parse_json_object(raw_text: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw_text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {"_parse_error": True, "_raw": raw_text}


def _field_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True


def load_prediction_cache(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row["document_id"]),
            str(row["model_id"]),
            str(row["task"]),
            str(row["modality"]),
        )
        cache[key] = row
    return cache


def append_prediction(path: Path, row: PredictionRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def _dry_run_prediction(task: str, sample: dict[str, Any]) -> tuple[Any, str]:
    if task == "classify":
        label = str(sample.get("label") or RVL_LABELS[0])
        return label, label
    payload = {
        "document_type_guess": str(sample.get("label") or ""),
        "title_or_subject": f"[dry-run] {sample.get('document_id')}",
        "parties": [],
        "dates": [],
        "identifiers": {},
        "amounts": {},
        "addresses_or_locations": [],
        "key_value_fields": {},
        "summary": "Dry-run placeholder annotation.",
        "extraction_notes": "DRY_RUN=True; no OpenRouter call.",
    }
    return payload, json.dumps(payload, ensure_ascii=False)


def run_one(
    *,
    sample: dict[str, Any],
    task: str,
    modality: str,
    model_id: str,
    image_path: Path | None,
    ocr_text: str | None,
    client: OpenRouterClient | None,
    dry_run: bool,
) -> PredictionRow:
    base = PredictionRow(
        document_id=str(sample["document_id"]),
        label_id=int(sample["label_id"]),
        label=str(sample["label"]),
        task=task,
        modality=modality,
        model_id=model_id,
        prediction=None,
        raw_text="",
        image_path=str(image_path) if image_path else None,
        dry_run=dry_run,
    )
    if dry_run:
        # Dry-run does not require images/OCR so docs + CI can score without the archive.
        pred, raw = _dry_run_prediction(task, sample)
        base.prediction = pred
        base.raw_text = raw
        base.latency_seconds = 0.0
        return base

    if modality == "vision" and image_path is None:
        base.error = "vision requires materialized image"
        return base
    if modality == "text" and not (ocr_text or "").strip():
        base.error = "text modality requires OCR text"
        return base

    assert client is not None
    labels = label_list_str()
    try:
        if task == "classify":
            if modality == "vision":
                prompt = load_prompt("rvl_classify_vision").format(label_list=labels)
                start = time.perf_counter()
                resp = client.complete_multimodal(
                    prompt,
                    image=image_path,
                    max_tokens=64,
                    system_prompt=SYSTEM_PROMPT,
                )
            else:
                prompt = load_prompt("rvl_classify_text").format(
                    label_list=labels,
                    document_text=(ocr_text or "")[:12000],
                )
                start = time.perf_counter()
                resp = client.complete(
                    prompt, max_tokens=64, system_prompt=SYSTEM_PROMPT
                )
            latency = time.perf_counter() - start
            raw = resp["text"]
            base.prediction = normalize_rvl_label(raw)
            base.raw_text = raw
        elif task == "extract_annotate":
            if modality == "vision":
                prompt = load_prompt("rvl_extract_annotate_vision")
                start = time.perf_counter()
                resp = client.complete_multimodal(
                    prompt,
                    image=image_path,
                    max_tokens=1024,
                    system_prompt=SYSTEM_PROMPT,
                )
            else:
                prompt = load_prompt("rvl_extract_annotate_text").format(
                    document_text=(ocr_text or "")[:12000]
                )
                start = time.perf_counter()
                resp = client.complete(
                    prompt, max_tokens=1024, system_prompt=SYSTEM_PROMPT
                )
            latency = time.perf_counter() - start
            raw = resp["text"]
            base.prediction = parse_json_object(raw)
            base.raw_text = raw
        else:
            raise ValueError(f"Unknown task {task}")

        usage = resp.get("usage") or {}
        base.input_tokens = int(usage.get("input_tokens") or 0)
        base.output_tokens = int(usage.get("output_tokens") or 0)
        base.latency_seconds = float(latency)
        base.model_id = str(resp.get("model") or model_id)
    except Exception as exc:  # noqa: BLE001 — record per-row failure
        base.error = str(exc)
    return base


def run_eval(
    samples: Sequence[dict[str, Any]],
    *,
    materialize: Sequence[MaterializeResult] | None = None,
    vision_models: Sequence[str] = DEFAULT_VISION_MODELS,
    text_models: Sequence[str] = DEFAULT_TEXT_MODELS,
    tasks: Sequence[str] = ("classify", "extract_annotate"),
    modalities: Sequence[str] = ("vision", "text"),
    out_dir: Path | None = None,
    dry_run: bool = True,
    cfg: Config | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the PoC eval loop and write prediction + summary artifacts."""
    out = Path(out_dir or DEFAULT_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"

    mat_by_id: dict[str, MaterializeResult] = {}
    if materialize is not None:
        mat_by_id = {m.document_id: m for m in materialize}
    else:
        need_vision = "vision" in modalities
        need_text = "text" in modalities
        if need_vision or need_text:
            mats = materialize_samples(list(samples), run_ocr=need_text)
            mat_by_id = {m.document_id: m for m in mats}

    cache = load_prediction_cache(pred_path) if resume else {}
    config = cfg or Config.load()
    clients: dict[str, OpenRouterClient] = {}

    def get_client(model: str) -> OpenRouterClient:
        if model not in clients:
            clients[model] = OpenRouterClient(model=model, cfg=config)
        return clients[model]

    new_rows: list[PredictionRow] = []
    for sample in samples:
        mid = mat_by_id.get(str(sample["document_id"]))
        image_path = mid.image_path if mid else None
        ocr_text = mid.ocr_text if mid else None

        for modality in modalities:
            models = vision_models if modality == "vision" else text_models
            for model_id in models:
                for task in tasks:
                    key = (str(sample["document_id"]), model_id, task, modality)
                    if key in cache:
                        cached = cache[key]
                        row = PredictionRow(
                            document_id=cached["document_id"],
                            label_id=int(cached["label_id"]),
                            label=cached["label"],
                            task=cached["task"],
                            modality=cached["modality"],
                            model_id=cached["model_id"],
                            prediction=cached.get("prediction"),
                            raw_text=cached.get("raw_text") or "",
                            input_tokens=int(cached.get("input_tokens") or 0),
                            output_tokens=int(cached.get("output_tokens") or 0),
                            latency_seconds=float(cached.get("latency_seconds") or 0),
                            cost_usd=float(cached.get("cost_usd") or 0),
                            error=cached.get("error"),
                            dry_run=bool(cached.get("dry_run")),
                            image_path=cached.get("image_path"),
                            cache_hit=True,
                        )
                        new_rows.append(row)
                        continue

                    client = None if dry_run else get_client(model_id)
                    row = run_one(
                        sample=sample,
                        task=task,
                        modality=modality,
                        model_id=model_id,
                        image_path=image_path,
                        ocr_text=ocr_text,
                        client=client,
                        dry_run=dry_run,
                    )
                    append_prediction(pred_path, row)
                    cache[key] = row.to_dict()
                    new_rows.append(row)

    # Reload full cache for scoring (includes prior resume rows)
    all_cached = load_prediction_cache(pred_path)
    all_rows = list(all_cached.values())
    # Prefer in-memory rows for this run's keys when scoring current slice
    scored_source = [r.to_dict() for r in new_rows] if new_rows else all_rows

    cls_summary = score_classification(scored_source)
    ext_summary = score_extraction(scored_source)
    mat_summary = summarize_materialize(mat_by_id.values()) if mat_by_id else {}

    manifest = {
        "experiment": "rvl_cdip_openrouter_eval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "vision_models": list(vision_models),
        "text_models": list(text_models),
        "tasks": list(tasks),
        "modalities": list(modalities),
        "dry_run": dry_run,
        "predictions_path": str(pred_path.relative_to(REPO_ROOT))
        if pred_path.is_relative_to(REPO_ROOT)
        else str(pred_path),
        "materialize": mat_summary,
        "classification": cls_summary,
        "extraction": ext_summary,
    }

    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_json_csv(out / "summary_classification", cls_summary)
    _write_json_csv(out / "summary_extraction", ext_summary)
    return manifest


def score_classification(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task") != "classify":
            continue
        groups[(str(row["model_id"]), str(row["modality"]))].append(row)

    per_model: list[dict[str, Any]] = []
    for (model_id, modality), group in sorted(groups.items()):
        usable = [r for r in group if not r.get("error")]
        n = len(usable)
        if n == 0:
            per_model.append(
                {
                    "model_id": model_id,
                    "modality": modality,
                    "n": 0,
                    "accuracy": None,
                    "error_rate": 1.0,
                }
            )
            continue
        correct = 0
        per_class_tot: Counter[str] = Counter()
        per_class_hit: Counter[str] = Counter()
        for r in usable:
            gold = str(r["label"])
            pred = normalize_rvl_label(r.get("prediction"))
            per_class_tot[gold] += 1
            if pred == gold:
                correct += 1
                per_class_hit[gold] += 1
        recalls = {
            lab: (per_class_hit[lab] / per_class_tot[lab] if per_class_tot[lab] else 0.0)
            for lab in RVL_LABELS
            if per_class_tot[lab]
        }
        macro_recall = sum(recalls.values()) / len(recalls) if recalls else 0.0
        per_model.append(
            {
                "model_id": model_id,
                "modality": modality,
                "n": n,
                "accuracy": correct / n,
                "macro_recall": macro_recall,
                "per_class_recall": recalls,
                "error_rate": sum(1 for r in group if r.get("error")) / max(len(group), 1),
                "avg_latency_seconds": sum(float(r.get("latency_seconds") or 0) for r in usable)
                / n,
                "total_input_tokens": sum(int(r.get("input_tokens") or 0) for r in usable),
                "total_output_tokens": sum(int(r.get("output_tokens") or 0) for r in usable),
            }
        )
    return {"per_model": per_model}


def score_extraction(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task") != "extract_annotate":
            continue
        groups[(str(row["model_id"]), str(row["modality"]))].append(row)

    per_model: list[dict[str, Any]] = []
    preds_by_doc: dict[str, dict[str, Any]] = {}

    for (model_id, modality), group in sorted(groups.items()):
        usable = [r for r in group if not r.get("error")]
        n = len(usable)
        valid = 0
        fill_counts = Counter({k: 0 for k in EXTRACT_KEYS})
        for r in usable:
            pred = r.get("prediction")
            if isinstance(pred, dict) and not pred.get("_parse_error"):
                valid += 1
                for key in EXTRACT_KEYS:
                    if _field_filled(pred.get(key)):
                        fill_counts[key] += 1
                preds_by_doc.setdefault(str(r["document_id"]), {})[
                    f"{model_id}|{modality}"
                ] = pred
        fill_rates = {k: (fill_counts[k] / n if n else 0.0) for k in EXTRACT_KEYS}
        per_model.append(
            {
                "model_id": model_id,
                "modality": modality,
                "n": n,
                "valid_json_rate": valid / n if n else None,
                "mean_field_fill_rate": (
                    sum(fill_rates.values()) / len(fill_rates) if n else None
                ),
                "field_fill_rates": fill_rates,
                "error_rate": sum(1 for r in group if r.get("error")) / max(len(group), 1),
            }
        )

    agreement = _pairwise_field_agreement(preds_by_doc)
    return {"per_model": per_model, "pairwise_agreement": agreement}


def _pairwise_field_agreement(
    preds_by_doc: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Share of docs where two model keys agree on filled scalar/list fields."""
    model_keys = sorted({mk for d in preds_by_doc.values() for mk in d})
    if len(model_keys) < 2:
        return {"n_pairs": 0, "pairs": []}

    pairs_out: list[dict[str, Any]] = []
    for i, a in enumerate(model_keys):
        for b in model_keys[i + 1 :]:
            compared = 0
            agreed = 0
            for _doc, models in preds_by_doc.items():
                if a not in models or b not in models:
                    continue
                pa, pb = models[a], models[b]
                for key in ("document_type_guess", "title_or_subject", "summary"):
                    va, vb = pa.get(key), pb.get(key)
                    if not (_field_filled(va) or _field_filled(vb)):
                        continue
                    compared += 1
                    if normalize_label(va) == normalize_label(vb):
                        agreed += 1
            pairs_out.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "n_compared_fields": compared,
                    "agreement_rate": agreed / compared if compared else None,
                }
            )
    return {"n_pairs": len(pairs_out), "pairs": pairs_out}


def _write_json_csv(stem: Path, payload: dict[str, Any]) -> None:
    stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = payload.get("per_model") or []
    if not rows:
        return
    # Flatten nested dicts for CSV
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat_rows.append(flat)
    if not flat_rows:
        return
    keys = list(flat_rows[0].keys())
    with stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)
