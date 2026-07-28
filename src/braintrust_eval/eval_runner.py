"""Braintrust Eval runner for RVL-CDIP vision classification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from src.braintrust_eval.classifier import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_VISION_REASONING_MODEL,
    ClassificationResult,
    classify_image,
    render_classification_prompt,
)
from src.braintrust_eval.dataset import load_samples
from src.rvl_cdip.openrouter_eval import normalize_rvl_label
from src.utils.config import Config, REPO_ROOT
from src.utils.io import write_json


def exact_match_scorer(input: Any, output: Any, expected: Any) -> float:
    """Braintrust scorer: 1.0 if normalized prediction equals gold label."""
    pred = output
    if isinstance(output, dict):
        pred = output.get("prediction") or output.get("label") or ""
    return (
        1.0
        if normalize_rvl_label(pred) == normalize_rvl_label(expected)
        else 0.0
    )


def _attachment_to_temp_png(attachment: Any) -> Path:
    """Materialize a Braintrust Attachment / ReadonlyAttachment to a temp PNG."""
    data = getattr(attachment, "data", None)
    if callable(data):
        payload = data()
    else:
        payload = data
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    elif isinstance(payload, str) and Path(payload).is_file():
        raw = Path(payload).read_bytes()
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raise TypeError(f"Unsupported attachment payload type: {type(payload)!r}")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _local_eval_cases(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in samples:
        cases.append(
            {
                "input": {
                    "document_id": row["document_id"],
                    "label_id": int(row["label_id"]),
                    "fixed_image_path": row["fixed_image_path"],
                },
                "expected": row["label"],
                "metadata": {
                    "label": row["label"],
                    "label_id": int(row["label_id"]),
                    "placeholder": bool(row.get("placeholder")),
                },
            }
        )
    return cases


def run_local_loop(
    *,
    samples: Sequence[dict[str, Any]] | None = None,
    model_id: str = DEFAULT_VISION_REASONING_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    dry_run: bool = True,
    cfg: Config | None = None,
    prompt_template: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run classify without Braintrust (local JSONL scoreboard + cost stats)."""
    rows = list(samples) if samples is not None else load_samples()
    config = cfg or Config.load()
    results: list[ClassificationResult] = []
    for row in rows:
        results.append(
            classify_image(
                image_path=Path(str(row["fixed_image_path"])),
                expected_label=str(row["label"]),
                document_id=str(row["document_id"]),
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_template=prompt_template,
                cfg=config,
                dry_run=dry_run,
            )
        )

    n = len(results)
    n_ok = sum(1 for r in results if r.exact_match)
    n_err = sum(1 for r in results if r.error)
    scored = [r for r in results if not r.error]
    avg = lambda key: (  # noqa: E731
        sum(getattr(r, key) for r in scored) / len(scored) if scored else 0.0
    )
    summary = {
        "mode": "local",
        "model_id": model_id,
        "n": n,
        "accuracy_exact_match": (n_ok / n) if n else 0.0,
        "n_correct": n_ok,
        "n_errors": n_err,
        "dry_run": dry_run,
        "avg_prompt_tokens": avg("input_tokens"),
        "avg_completion_tokens": avg("output_tokens"),
        "avg_prompt_cached_tokens": avg("prompt_cached_tokens"),
        "avg_completion_reasoning_tokens": avg("completion_reasoning_tokens"),
        "avg_total_tokens": avg("total_tokens"),
        "avg_latency_seconds": avg("latency_seconds"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt_preview": render_classification_prompt(prompt_template=prompt_template)[
            :400
        ],
    }

    dest = Path(out_dir or (REPO_ROOT / "data" / "braintrust" / "runs"))
    dest.mkdir(parents=True, exist_ok=True)
    pred_path = dest / f"predictions_{model_id.replace('/', '_')}.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    summary_path = dest / f"summary_{model_id.replace('/', '_')}.json"
    write_json(summary_path, summary)
    summary["predictions_path"] = str(pred_path)
    summary["summary_path"] = str(summary_path)
    return summary


def run_braintrust_eval(
    *,
    model_id: str = DEFAULT_VISION_REASONING_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    dry_run: bool = False,
    use_remote_dataset: bool = True,
    experiment_name: str | None = None,
    prompt_template: str | None = None,
    cfg: Config | None = None,
    max_concurrency: int | None = None,
) -> Any:
    """Execute a Braintrust ``Eval`` against the fixed-size sampled dataset.

    When ``use_remote_dataset=True``, loads the Braintrust-hosted dataset
    (must upload first). Otherwise evaluates local ``samples.jsonl`` rows and
    still logs the experiment to Braintrust.
    """
    try:
        from braintrust import Eval, init_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'braintrust is not installed. Run: pip install -e ".[braintrust]"'
        ) from exc

    config = cfg or Config.load()
    if not config.braintrust_api_key and not dry_run:
        # Braintrust can still run offline-ish, but we require the key for real logs.
        raise RuntimeError("BRAINTRUST_API_KEY is not set.")

    project = config.braintrust_project
    org = config.braintrust_org or None

    if use_remote_dataset:
        data: Any = init_dataset(
            project=project,
            name=config.braintrust_dataset,
            org_name=org,
            api_key=config.braintrust_api_key or None,
        )
    else:
        data = _local_eval_cases(load_samples())

    def task(input: dict[str, Any], hooks: Any = None) -> dict[str, Any]:
        image_path: Path | None = None
        temp_path: Path | None = None
        try:
            if "fixed_image_path" in input and input["fixed_image_path"]:
                image_path = Path(str(input["fixed_image_path"]))
            elif "image" in input and input["image"] is not None:
                temp_path = _attachment_to_temp_png(input["image"])
                image_path = temp_path
            else:
                raise ValueError("Eval case missing image / fixed_image_path")

            expected = ""
            if hooks is not None:
                expected = str(getattr(hooks, "expected", None) or "")
            if not expected:
                expected = str(
                    input.get("label")
                    or (input.get("metadata") or {}).get("label")
                    or "unknown"
                )

            result = classify_image(
                image_path=image_path,
                expected_label=expected,
                document_id=str(input.get("document_id") or image_path.stem),
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_template=prompt_template,
                cfg=config,
                dry_run=dry_run,
            )
            if hooks is not None and hasattr(hooks, "meta"):
                try:
                    hooks.meta(
                        {
                            "usage": {
                                "prompt_tokens": result.input_tokens,
                                "completion_tokens": result.output_tokens,
                                "prompt_cached_tokens": result.prompt_cached_tokens,
                                "completion_reasoning_tokens": (
                                    result.completion_reasoning_tokens
                                ),
                            },
                            "reasoning_preview": (result.reasoning or "")[:2000],
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
            return {
                "prediction": result.prediction,
                "raw_text": result.raw_text,
                "reasoning": result.reasoning,
                "usage": {
                    "prompt_tokens": result.input_tokens,
                    "completion_tokens": result.output_tokens,
                    "prompt_cached_tokens": result.prompt_cached_tokens,
                    "completion_reasoning_tokens": result.completion_reasoning_tokens,
                    "total_tokens": result.total_tokens,
                },
                "latency_seconds": result.latency_seconds,
                "error": result.error,
                "model_id": result.model_id,
            }
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    exp_name = experiment_name or f"{model_id.replace('/', '_')}"
    return Eval(
        project,
        data=data,
        task=task,
        scores=[exact_match_scorer],
        experiment_name=exp_name,
        metadata={
            "model_id": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "dry_run": dry_run,
            "image_size": 1024,
            "prompt": "CLASSIFICATION_PROMPT / rvl_classify_vision_bt",
        },
        max_concurrency=max_concurrency or config.max_concurrency,
    )


def build_task_for_tests(
    classify_fn: Callable[..., ClassificationResult],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Test helper — wrap a classify function as an Eval task."""

    def task(input: dict[str, Any]) -> dict[str, Any]:
        result = classify_fn(
            image_path=Path(str(input["fixed_image_path"])),
            expected_label=str(input.get("label") or "unknown"),
            document_id=str(input.get("document_id") or ""),
        )
        return {"prediction": result.prediction, "reasoning": result.reasoning}

    return task
