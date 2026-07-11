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

logger = logging.getLogger(__name__)


def run_seed(n: int = 240, seed: int = 42, skip_ingest: bool = True) -> dict[str, str]:
    cfg = Config.load()
    if not skip_ingest:
        from src.generation.corpus_ingest import run_ingest

        run_ingest(cfg, funsd_n=20, doclaynet_n=10, rvl_n=2)
    run_profiler(cfg)
    sk_path = run_sampler(cfg, n=n, seed=seed)
    doc_path = run_stage_a(cfg, sk_path, seed=seed)
    memo_path = run_stage_b(cfg, doc_path, seed=seed)
    noisy_path = run_noise(cfg, doc_path, seed=seed)
    return {
        "skeletons": str(sk_path),
        "documents": str(doc_path),
        "memos": str(memo_path),
        "noisy": str(noisy_path),
        "splits": str(cfg.splits_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ingest", action="store_true", help="Also attempt Hub corpus ingest")
    args = parser.parse_args()
    paths = run_seed(n=args.n, seed=args.seed, skip_ingest=not args.ingest)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
