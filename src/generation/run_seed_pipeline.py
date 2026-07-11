"""End-to-end seed corpus generation for Phases 1–3."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.generation.characteristic_profiler import run_profiler
from src.generation.noise_injection import run_noise
from src.generation.skeleton_sampler import run_sampler
from src.generation.stage_a_document_gen import run_stage_a
from src.generation.stage_b_memo_gen import run_stage_b
from src.utils.config import Config
from src.utils.wandb_utils import add_wandb_cli_flags, settings_from_args, start_run

logger = logging.getLogger(__name__)


def run_seed(
    n: int = 240,
    seed: int = 42,
    skip_ingest: bool = True,
    wandb_settings=None,
    wandb_run_name: str | None = None,
) -> dict[str, str]:
    cfg = Config.load()
    run_name = wandb_run_name or f"seed-pipeline-n{n}-seed{seed}"
    with start_run(
        name=run_name,
        job_type="generation",
        config={
            "task": "seed_pipeline",
            "n": n,
            "seed": seed,
            "ingest": not skip_ingest,
        },
        tags=["generation", "seed"],
        settings=wandb_settings,
    ) as wb:
        if not skip_ingest:
            from src.generation.corpus_ingest import run_ingest

            run_ingest(cfg, funsd_n=20, doclaynet_n=10, rvl_n=2)
            wb.log({"progress/ingest_done": 1})
        run_profiler(cfg)
        wb.log({"progress/profiler_done": 1})
        sk_path = run_sampler(cfg, n=n, seed=seed)
        wb.log({"progress/skeletons_done": 1, "n_requested": n})
        doc_path = run_stage_a(cfg, sk_path, seed=seed)
        wb.log({"progress/stage_a_done": 1})
        memo_path = run_stage_b(cfg, doc_path, seed=seed)
        wb.log({"progress/stage_b_done": 1})
        noisy_path = run_noise(cfg, doc_path, seed=seed)
        wb.log({"progress/noise_done": 1})
        paths = {
            "skeletons": str(sk_path),
            "documents": str(doc_path),
            "memos": str(memo_path),
            "noisy": str(noisy_path),
            "splits": str(cfg.splits_path),
        }
        wb.summary({**paths, "n": n, "seed": seed})
        artifact_paths = [Path(p) for p in paths.values() if Path(p).is_file()]
        if artifact_paths:
            wb.log_artifact_files(
                name=f"seed-pipeline-outputs-n{n}-seed{seed}",
                paths=artifact_paths[:5],
                artifact_type="dataset",
                metadata={"n": n, "seed": seed},
            )
        return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ingest", action="store_true", help="Also attempt Hub corpus ingest")
    add_wandb_cli_flags(parser)
    args = parser.parse_args()
    paths = run_seed(
        n=args.n,
        seed=args.seed,
        skip_ingest=not args.ingest,
        wandb_settings=settings_from_args(args),
        wandb_run_name=args.wandb_run_name,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
