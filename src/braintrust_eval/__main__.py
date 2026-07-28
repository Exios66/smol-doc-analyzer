"""CLI for Braintrust RVL-CDIP fixed-size classification experiments.

Examples::

  # 1) Build 10×16 @ 1024² sample set (needs RVL index + images)
  python -m src.braintrust_eval build-dataset

  # CI / wiring without the 38 GB archive:
  python -m src.braintrust_eval build-dataset --placeholder

  # 2) Upload to Braintrust (needs BRAINTRUST_API_KEY)
  python -m src.braintrust_eval upload-dataset

  # 3) Local dry-run scoreboard
  python -m src.braintrust_eval run-eval --dry-run --local

  # 4) Live Kimi K3 Braintrust experiment (captures reasoning)
  python -m src.braintrust_eval run-eval --model moonshotai/kimi-k3

  # 5) Cost projections (Kimi observed avgs → Sonnet/Opus)
  python -m src.braintrust_eval cost-estimate

  # 6) Improve prompt from misclassifications via DeepSeek R1
  python -m src.braintrust_eval improve-prompt \\
    --predictions data/braintrust/runs/predictions_moonshotai_kimi-k3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.braintrust_eval.classifier import (
    DEFAULT_FLAGSHIP_MODELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT_IMPROVE_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_VISION_REASONING_MODEL,
)
from src.braintrust_eval.cost_estimate import project_scale_table
from src.braintrust_eval.dataset import (
    DEFAULT_N_PER_CLASS,
    DEFAULT_OUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SIZE,
    build_fixed_size_sampled,
)
from src.utils.config import Config


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    manifest = build_fixed_size_sampled(
        out_dir=Path(args.out_dir) if args.out_dir else None,
        n_per_class=args.n_per_class,
        seed=args.seed,
        size=args.size,
        split=args.split,
        materialize=not args.no_materialize,
        placeholder=args.placeholder,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_upload_dataset(args: argparse.Namespace) -> int:
    from src.braintrust_eval.upload_dataset import upload_dataset

    result = upload_dataset(
        project=args.project,
        dataset_name=args.dataset,
        org_name=args.org,
        clear_existing=args.clear,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_run_eval(args: argparse.Namespace) -> int:
    from src.braintrust_eval.eval_runner import run_braintrust_eval, run_local_loop

    if args.local:
        summary = run_local_loop(
            model_id=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            dry_run=args.dry_run,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )
        print(json.dumps(summary, indent=2))
        return 0

    result = run_braintrust_eval(
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        dry_run=args.dry_run,
        use_remote_dataset=not args.local_cases,
        experiment_name=args.experiment,
    )
    # Eval returns a rich object; print a compact summary when available.
    summary = getattr(result, "summary", None)
    if callable(summary):
        print(summary())
    else:
        print(result)
    return 0


def _cmd_cost_estimate(args: argparse.Namespace) -> int:
    models = args.model or [
        DEFAULT_VISION_REASONING_MODEL,
        *DEFAULT_FLAGSHIP_MODELS,
    ]
    rows = project_scale_table(model_ids=models, counts=args.counts)
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_improve_prompt(args: argparse.Namespace) -> int:
    from src.braintrust_eval.prompt_improve import improve_prompt
    from src.utils.io import write_json

    result = improve_prompt(
        predictions_path=Path(args.predictions),
        model_id=args.model,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    out = Path(args.out) if args.out else (
        Path("data/braintrust/runs") / "improved_prompt.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    prompt_out = out.with_suffix(".txt")
    prompt_out.write_text(str(result.get("improved_prompt") or ""), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "improved_prompt"}, indent=2))
    print(f"wrote prompt → {prompt_out}")
    print(f"wrote meta   → {out}")
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    cfg = Config.load()
    status = {
        "OPENROUTER_API_KEY": bool(cfg.openrouter_api_key),
        "BRAINTRUST_API_KEY": bool(cfg.braintrust_api_key),
        "braintrust_project": cfg.braintrust_project,
        "braintrust_dataset": cfg.braintrust_dataset,
        "braintrust_org": cfg.braintrust_org,
        "local_samples": (DEFAULT_OUT_DIR / "samples.jsonl").is_file(),
        "default_vision_model": DEFAULT_VISION_REASONING_MODEL,
        "default_prompt_improve_model": DEFAULT_PROMPT_IMPROVE_MODEL,
    }
    print(json.dumps(status, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.braintrust_eval",
        description="Braintrust RVL-CDIP fixed-size vision classification experiments",
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-dataset", help="Build 1024×1024 padded sample set")
    b.add_argument("--out-dir", type=str, default=None)
    b.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--size", type=int, default=DEFAULT_SIZE)
    b.add_argument("--split", type=str, default="test")
    b.add_argument(
        "--placeholder",
        action="store_true",
        help="Write synthetic PNGs (no RVL archive required)",
    )
    b.add_argument(
        "--no-materialize",
        action="store_true",
        help="Skip selective extract from rvl-cdip.tar.gz",
    )
    b.set_defaults(func=_cmd_build_dataset)

    u = sub.add_parser("upload-dataset", help="Upload sample set to Braintrust")
    u.add_argument("--project", type=str, default=None)
    u.add_argument("--dataset", type=str, default=None)
    u.add_argument("--org", type=str, default=None)
    u.add_argument("--clear", action="store_true", help="Clear dataset before insert")
    u.set_defaults(func=_cmd_upload_dataset)

    r = sub.add_parser("run-eval", help="Run classification eval (local or Braintrust)")
    r.add_argument("--model", type=str, default=DEFAULT_VISION_REASONING_MODEL)
    r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    r.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument(
        "--local",
        action="store_true",
        help="Score locally to JSONL (no Braintrust experiment)",
    )
    r.add_argument(
        "--local-cases",
        action="store_true",
        help="Use local samples.jsonl as Eval data but still log to Braintrust",
    )
    r.add_argument("--experiment", type=str, default=None)
    r.add_argument("--out-dir", type=str, default=None)
    r.set_defaults(func=_cmd_run_eval)

    c = sub.add_parser("cost-estimate", help="Project USD cost at scale")
    c.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model slug (repeatable). Default: kimi + sonnet + opus",
    )
    c.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[160, 800, 25_000, 320_000],
    )
    c.set_defaults(func=_cmd_cost_estimate)

    i = sub.add_parser(
        "improve-prompt",
        help="Use DeepSeek R1 to revise CLASSIFICATION_PROMPT from errors",
    )
    i.add_argument("--predictions", type=str, required=True)
    i.add_argument("--model", type=str, default=DEFAULT_PROMPT_IMPROVE_MODEL)
    i.add_argument("--limit", type=int, default=24)
    i.add_argument("--out", type=str, default=None)
    i.add_argument("--dry-run", action="store_true")
    i.set_defaults(func=_cmd_improve_prompt)

    s = sub.add_parser("status", help="Show key + project readiness (no secrets)")
    s.set_defaults(func=_cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
