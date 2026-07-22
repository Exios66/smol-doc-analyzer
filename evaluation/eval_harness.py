"""
eval_harness.py

Runs the same evaluation set through frontier models (Anthropic + OpenAI, via
OpenRouter) and local fine-tuned models (DeBERTa-v3 classifier, LayoutLM/Donut
extractor, LoRA-tuned generative memo model) across three tasks:

    1. classification   -- document -> ACORD taxonomy label
    2. extraction        -- document -> structured claim fields
    3. memo_generation    -- document/skeleton -> adjuster memo

For every (task, backend, example) triple it records: prediction, ground
truth, latency, token counts, and computed cost -- then writes one row per
result to a JSONL log (source of truth) and a flattened CSV (feeds the cost
model spreadsheet).

Usage:
    python -m evaluation.eval_harness \\
        --eval-set data/eval/eval_set.jsonl \\
        --tasks classification extraction memo_generation \\
        --backends anthropic openai local \\
        --n-samples 50 \\
        --output-dir evaluation/results/eval_run_2026-07-13 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from evaluation.cost_model_helpers import local_cost_per_call, load_pricing_raw
from evaluation.local_backends import build_local_task_endpoints
from evaluation.metrics import normalize_label
from src.utils.config import Config
from src.utils.llm_client import OpenRouterClient, is_free_model
from src.utils.prompts import load_prompt
from src.utils.provenance import ProvenanceLogger

PRICING_CONFIG_PATH = Path(__file__).parent / "pricing.yaml"


@dataclass
class ModelPricing:
    input_per_million: float  # USD per 1M input tokens
    output_per_million: float  # USD per 1M output tokens


@dataclass
class EvalResult:
    run_id: str
    task: str
    backend: str
    model_id: str
    example_id: str
    prediction: Any
    ground_truth: Any
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float
    correct: bool | None = None  # filled in by metrics pass
    score: float | None = None  # filled in by metrics pass (extraction F1, judge score, etc.)
    error: str | None = None


def load_pricing(path: Path | None = None) -> tuple[dict[str, ModelPricing], float]:
    """Returns (frontier model pricing by name, local GPU hourly rate USD).

    Creates ``evaluation/pricing.yaml`` from defaults when the file is missing.
    """
    raw = load_pricing_raw(path or PRICING_CONFIG_PATH, create_if_missing=True)
    frontier = {
        k: ModelPricing(
            input_per_million=float(v["input_per_million"]),
            output_per_million=float(v["output_per_million"]),
        )
        for k, v in raw["frontier_models"].items()
    }
    local_gpu_hourly_rate = float(raw["local_compute"]["gpu_hourly_rate_usd"])
    return frontier, local_gpu_hourly_rate


def compute_cost(pricing: ModelPricing, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * pricing.input_per_million + (
        output_tokens / 1_000_000
    ) * pricing.output_per_million


# --- backends ----------------------------------------------------------------
# Each backend implements run(task, example) -> (prediction, input_tokens, output_tokens, latency)


class FrontierBackend:
    """Wraps a single OpenRouter-routed frontier model (Anthropic or OpenAI)."""

    def __init__(self, name: str, model_slug: str, pricing: ModelPricing, cfg: Config | None = None):
        self.name = name
        self.model_slug = model_slug
        self.pricing = pricing
        self.client = OpenRouterClient(model=model_slug, cfg=cfg)

    def run(self, task: str, example: dict) -> tuple[Any, int, int, float, str]:
        prompt_template = load_prompt(f"eval_{task}")
        prompt = prompt_template.format(**_stringify_prompt_fields(example["prompt_fields"]))

        start = time.perf_counter()
        response = self.client.complete(prompt, max_tokens=example.get("max_tokens", 1024))
        latency = time.perf_counter() - start

        prediction = parse_prediction(task, response["text"])
        used_model = str(response.get("model") or self.model_slug)
        return (
            prediction,
            response["usage"]["input_tokens"],
            response["usage"]["output_tokens"],
            latency,
            used_model,
        )


def _stringify_prompt_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Ensure nested dict/list values are safe for ``str.format``."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


class LocalBackend:
    """
    Dispatches to whichever local fine-tuned model handles the task:
      - classification  -> DeBERTa-v3 encoder head
      - extraction       -> LayoutLMv3/Donut
      - memo_generation    -> LoRA-tuned Qwen2.5/Llama-3.1 (or template fallback)

    Cost is computed from amortized GPU-hour rate rather than per-token
    pricing -- see pricing.yaml's ``local_compute`` block and
    ``cost_model_helpers.local_cost_per_call()``.
    """

    def __init__(self, gpu_hourly_rate: float, task_endpoints: dict[str, Callable]):
        self.gpu_hourly_rate = gpu_hourly_rate
        self.task_endpoints = task_endpoints  # task -> callable(example) -> pred
        self.model_slug = "local"

    def run(self, task: str, example: dict) -> tuple[Any, int, int, float, str]:
        endpoint = self.task_endpoints.get(task)
        if endpoint is None:
            raise ValueError(f"No local endpoint wired for task '{task}'")

        start = time.perf_counter()
        prediction = endpoint(example)
        latency = time.perf_counter() - start

        # Local models don't bill per-token; report 0/0 here and let the
        # cost model derive $/doc from gpu_hourly_rate * latency instead.
        return prediction, 0, 0, latency, self.model_slug


def parse_prediction(task: str, raw_text: str) -> Any:
    """Task-specific parsing of a frontier model's raw text into a comparable structure."""
    if task == "classification":
        return normalize_label(raw_text)
    if task == "extraction":
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Tolerate fenced JSON / leading prose from chat models.
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw_text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {"_parse_error": True, "_raw": raw_text}
    if task == "memo_generation":
        return raw_text.strip()
    raise ValueError(f"Unknown task '{task}'")


# --- orchestration -----------------------------------------------------------


def load_eval_set(path: Path, n_samples: int | None) -> list[dict]:
    if n_samples is not None and n_samples < 0:
        raise ValueError(f"--n-samples must be >= 0, got {n_samples}")
    examples = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if n_samples is not None:
        # Cap per task so --n-samples is a cost-control knob, not a global slice.
        by_task: dict[str, list[dict]] = {}
        for ex in examples:
            by_task.setdefault(ex["task"], []).append(ex)
        capped: list[dict] = []
        for rows in by_task.values():
            capped.extend(rows[:n_samples])
        return capped
    return examples


def run_eval(
    tasks: list[str],
    backends: dict[str, FrontierBackend | LocalBackend],
    eval_set: list[dict],
    frontier_pricing: dict[str, ModelPricing],
    run_id: str,
    dry_run: bool,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    for task in tasks:
        task_examples = [ex for ex in eval_set if ex["task"] == task]
        for backend_name, backend in backends.items():
            for ex in task_examples:
                if dry_run:
                    print(f"[dry-run] would call {backend_name} for {task}/{ex['example_id']}")
                    continue
                start = time.perf_counter()
                try:
                    prediction, in_tok, out_tok, latency, used_model = backend.run(task, ex)
                    # Prefer harness wall time so failures and successes share one clock.
                    wall = time.perf_counter() - start
                    latency = max(float(latency), wall)
                    model_id = str(used_model or getattr(backend, "model_slug", "local"))
                    if isinstance(backend, FrontierBackend):
                        if is_free_model(model_id):
                            # Free OpenRouter routes must not inherit paid slug pricing.
                            cost = 0.0
                        else:
                            cost = compute_cost(
                                frontier_pricing.get(backend_name, backend.pricing),
                                in_tok,
                                out_tok,
                            )
                    else:
                        cost = local_cost_per_call(backend.gpu_hourly_rate, latency)
                    parse_error = (
                        isinstance(prediction, dict) and prediction.get("_parse_error") is True
                    )
                    result = EvalResult(
                        run_id=run_id,
                        task=task,
                        backend=backend_name,
                        model_id=model_id,
                        example_id=ex["example_id"],
                        prediction=prediction,
                        ground_truth=ex["ground_truth"],
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        latency_seconds=latency,
                        cost_usd=cost,
                        error="prediction_parse_error" if parse_error else None,
                    )
                except Exception as e:  # noqa: BLE001 -- eval harness must not die mid-run
                    latency = time.perf_counter() - start
                    if isinstance(backend, LocalBackend):
                        cost = local_cost_per_call(backend.gpu_hourly_rate, latency)
                    else:
                        cost = 0.0
                    result = EvalResult(
                        run_id=run_id,
                        task=task,
                        backend=backend_name,
                        model_id=getattr(backend, "model_slug", "local"),
                        example_id=ex["example_id"],
                        prediction=None,
                        ground_truth=ex["ground_truth"],
                        input_tokens=0,
                        output_tokens=0,
                        latency_seconds=latency,
                        cost_usd=cost,
                        error=str(e),
                    )
                results.append(result)
    return results


def write_outputs(
    results: list[EvalResult], output_dir: Path, provenance: ProvenanceLogger
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "eval_results.jsonl"
    csv_path = output_dir / "eval_results.csv"

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in results:
            record = asdict(r)
            provenance.log(record)
            jf.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                row = asdict(r)
                # Flatten nested prediction/GT for spreadsheet consumers.
                if not isinstance(row["prediction"], (str, int, float, type(None), bool)):
                    row["prediction"] = json.dumps(row["prediction"], ensure_ascii=False)
                if not isinstance(row["ground_truth"], (str, int, float, type(None), bool)):
                    row["ground_truth"] = json.dumps(row["ground_truth"], ensure_ascii=False)
                writer.writerow(row)

    print(f"Wrote {len(results)} results -> {jsonl_path} / {csv_path}")


def build_backends(
    names: list[str],
    frontier_pricing: dict[str, ModelPricing],
    gpu_hourly_rate: float,
    cfg: Config | None = None,
) -> dict[str, FrontierBackend | LocalBackend]:
    cfg = cfg or Config.load()
    registry: dict[str, Callable[[], FrontierBackend | LocalBackend]] = {
        "anthropic": lambda: FrontierBackend(
            "anthropic",
            "anthropic/claude-sonnet-4.5",
            frontier_pricing["anthropic"],
            cfg=cfg,
        ),
        "openai": lambda: FrontierBackend(
            "openai",
            "openai/gpt-4o",
            frontier_pricing["openai"],
            cfg=cfg,
        ),
        "local": lambda: LocalBackend(
            gpu_hourly_rate=gpu_hourly_rate,
            task_endpoints=build_local_task_endpoints(cfg),
        ),
    }
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"Unknown backends: {unknown}. Choose from {sorted(registry)}")
    return {name: registry[name]() for name in names}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Frontier vs. local model eval harness")
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["classification", "extraction", "memo_generation"],
    )
    parser.add_argument("--backends", nargs="+", default=["anthropic", "openai", "local"])
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Cap examples per task (cost control)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned calls without spending money",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="After writing results, run evaluation.metrics and emit summary.csv/json",
    )
    args = parser.parse_args(argv)

    run_id = str(uuid.uuid4())[:8]
    cfg = Config.load()
    frontier_pricing, gpu_hourly_rate = load_pricing()
    backends = build_backends(args.backends, frontier_pricing, gpu_hourly_rate, cfg=cfg)
    eval_set = load_eval_set(args.eval_set, args.n_samples)
    provenance = ProvenanceLogger(
        run_id=run_id,
        tag="eval_comparison",
        log_path=cfg.provenance_log_path,
    )

    results = run_eval(
        args.tasks, backends, eval_set, frontier_pricing, run_id, args.dry_run
    )
    if not args.dry_run:
        write_outputs(results, args.output_dir, provenance)
        if args.score:
            from evaluation.metrics import score_results_file

            score_results_file(
                results_path=args.output_dir / "eval_results.jsonl",
                output_csv=args.output_dir / "summary.csv",
                output_json=args.output_dir / "summary.json",
            )


if __name__ == "__main__":
    main()
