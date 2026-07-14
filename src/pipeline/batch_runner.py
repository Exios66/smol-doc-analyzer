"""
Batch runner for the chained document-analysis pipeline.

Processes a document batch end-to-end without manual intervention and
surfaces confidence scores / human-review flags for low-certainty cases.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.pipeline.orchestrator import DocumentAnalysisOrchestrator
from src.pipeline.types import AnalysisDocument
from src.utils.config import Config
from src.utils.io import load_jsonl, write_json, write_jsonl
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)


def run_batch(
    documents: list[dict[str, Any]] | Path,
    out_dir: Path,
    cfg: Config | None = None,
    *,
    enable_vision: bool | None = None,
    classifier_dir: Path | None = None,
    extractor_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    cfg = cfg or Config.load()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(documents) if isinstance(documents, Path) else list(documents)
    if limit is not None:
        rows = rows[:limit]

    orch = DocumentAnalysisOrchestrator(
        cfg=cfg,
        classifier_dir=classifier_dir,
        extractor_dir=extractor_dir,
        enable_vision=enable_vision,
    )

    results: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for row in rows:
        ctx = orch.analyze(AnalysisDocument.from_row(row))
        payload = ctx.to_dict()
        results.append(payload)
        if payload.get("low_confidence") or any(
            f.startswith("low_confidence")
            or f.startswith("upstream_failed")
            or f.endswith("_failed")
            for f in payload.get("flags", [])
        ):
            review_queue.append(
                {
                    "record_id": payload["record_id"],
                    "claim_id": payload.get("claim_id"),
                    "flags": payload.get("flags"),
                    "classification": payload.get("classification"),
                    "extraction_fields": (payload.get("extraction") or {}).get("fields_flat"),
                }
            )

    results_path = out_dir / "batch_results.jsonl"
    review_path = out_dir / "human_review_queue.jsonl"
    summary_path = out_dir / "batch_summary.json"

    write_jsonl(results_path, results)
    write_jsonl(review_path, review_queue)
    summary = {
        "n": len(results),
        "chain": orch.stage_names,
        "n_review": len(review_queue),
        "review_rate": (len(review_queue) / len(results)) if results else 0.0,
        "results_path": str(results_path),
        "review_queue_path": str(review_path),
        "by_document_type": _count_by_type(results),
        "flag_counts": _count_flags(results),
    }
    write_json(summary_path, summary)
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"pipeline-batch-{out_dir.name}",
            stage="pipeline_batch",
            source=str(documents) if isinstance(documents, Path) else "in_memory",
            prompt_version="pipeline_v1",
            model="→".join(orch.stage_names),
            extra={
                "n": summary["n"],
                "n_review": summary["n_review"],
                "out_dir": str(out_dir),
            },
        ),
    )
    logger.info(
        "Batch complete: %d docs, %d flagged for review (%.0f%%)",
        summary["n"],
        summary["n_review"],
        100 * summary["review_rate"],
    )
    return summary


def _count_by_type(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        lab = ((r.get("classification") or {}).get("document_type")) or "unknown"
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def _count_flags(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        for flag in r.get("flags") or []:
            # Collapse heuristic fill flags
            key = flag.split(":")[0] if flag.startswith("extract_heuristic_fill:") else flag
            counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Batch-run the chained to_markdown→classify→extract→vision→summarize pipeline"
    )
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--classifier-dir", type=Path, default=None)
    parser.add_argument("--extractor-dir", type=Path, default=None)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()
    cfg = Config.load()
    out_dir = args.out_dir or (cfg.pipeline_output_dir / f"batch_{args.inp.stem}")

    if args.no_vision:
        enable_vision: bool | None = False
    elif args.vision:
        enable_vision = True
    else:
        enable_vision = None

    summary = run_batch(
        args.inp,
        out_dir,
        cfg=cfg,
        enable_vision=enable_vision,
        classifier_dir=args.classifier_dir,
        extractor_dir=args.extractor_dir,
        limit=args.limit,
    )
    print(summary["results_path"])
    print(f"review_queue={summary['n_review']}/{summary['n']}")


if __name__ == "__main__":
    main()
