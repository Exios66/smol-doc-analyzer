"""Prepare LayoutLMv3 token-classification datasets from rendered forms."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json

logger = logging.getLogger(__name__)


LABELS = [
    "O",
    "B-claim_id",
    "I-claim_id",
    "B-policy_number",
    "I-policy_number",
    "B-policyholder_name",
    "I-policyholder_name",
    "B-date_of_loss",
    "I-date_of_loss",
    "B-loss_type",
    "I-loss_type",
    "B-location",
    "I-location",
    "B-estimated_damage",
    "I-estimated_damage",
    "B-deductible",
    "I-deductible",
    "B-reserve_set",
    "I-reserve_set",
    "B-adjuster_assigned",
    "I-adjuster_assigned",
    "B-claimant",
    "I-claimant",
    "B-effective_date",
    "I-effective_date",
    "B-state",
    "I-state",
    "B-coverage_type",
    "I-coverage_type",
]


def assign_split(record_id: str, splits: dict[str, Any]) -> str:
    if record_id in splits.get("train", []):
        return "train"
    if record_id in splits.get("val", []):
        return "val"
    if record_id in splits.get("test", []):
        return "test"
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _normalize_box(bbox: list[int], width: int, height: int) -> list[int]:
    """Normalize pixel boxes to LayoutLMv3 0–1000 coordinate space."""
    w = max(int(width), 1)
    h = max(int(height), 1)
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, int(1000 * x0 / w))),
        max(0, min(1000, int(1000 * y0 / h))),
        max(0, min(1000, int(1000 * x1 / w))),
        max(0, min(1000, int(1000 * y1 / h))),
    ]


def prepare(rendered_path: Path, cfg: Config, out_dir: Path | None = None) -> Path:
    rows_in = load_jsonl(rendered_path)
    splits = read_json(cfg.splits_path) if cfg.splits_path.exists() else {}
    label2id = {l: i for i, l in enumerate(LABELS)}
    buckets = {"train": [], "val": [], "test": []}
    known = set(splits.get("train", []) + splits.get("val", []) + splits.get("test", []))
    miss = 0
    for row in rows_in:
        rid = row["record_id"]
        if known and rid not in known:
            miss += 1
        split = assign_split(rid, splits)
        words = row["words"]
        width = int(row.get("width") or row.get("page_width") or 0)
        height = int(row.get("height") or row.get("page_height") or 0)
        if (not width or not height) and words:
            # Infer page size from max pixel extents when render metadata is absent.
            width = max((w["bbox"][2] for w in words), default=1)
            height = max((w["bbox"][3] for w in words), default=1)
        width = max(width, 1)
        height = max(height, 1)
        buckets[split].append(
            {
                "record_id": rid,
                "image_path": row["image_path"],
                "tokens": [w["text"] for w in words],
                "bboxes": [_normalize_box(w["bbox"], width, height) for w in words],
                "labels": [label2id.get(w.get("label", "O"), 0) for w in words],
                "is_noisy": row.get("is_noisy", False),
                "width": width,
                "height": height,
                "truncated": bool(row.get("truncated")),
            }
        )
    if miss:
        rate = miss / max(len(rows_in), 1)
        logger.warning(
            "extraction prepare: %s/%s record_ids missing from splits (%.0f%%); "
            "used hash-stable fallback",
            miss,
            len(rows_in),
            100 * rate,
        )
        if rate > 0.5 and known:
            raise ValueError(
                f"Too many unmatched record_ids ({miss}/{len(rows_in)}). "
                "Regenerate skeletons/splits or align rendered record_ids with splits.json."
            )
    out = out_dir or (Path(rendered_path).parent / "extraction_prepared")
    out.mkdir(parents=True, exist_ok=True)
    for split, items in buckets.items():
        (out / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + ("\n" if items else ""),
            encoding="utf-8",
        )
    write_json(out / "label2id.json", label2id)
    write_json(
        out / "summary.json",
        {k: len(v) for k, v in buckets.items()} | {"split_misses": miss},
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = Config.load()
    print(prepare(args.inp, cfg, args.out))


if __name__ == "__main__":
    main()
