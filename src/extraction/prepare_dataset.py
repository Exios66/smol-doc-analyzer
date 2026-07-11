"""Prepare LayoutLMv3 token-classification datasets from rendered forms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json


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
    return "train"


def prepare(rendered_path: Path, cfg: Config, out_dir: Path | None = None) -> Path:
    rows_in = load_jsonl(rendered_path)
    splits = read_json(cfg.splits_path) if cfg.splits_path.exists() else {}
    label2id = {l: i for i, l in enumerate(LABELS)}
    buckets = {"train": [], "val": [], "test": []}
    for row in rows_in:
        split = assign_split(row["record_id"], splits)
        words = row["words"]
        buckets[split].append(
            {
                "record_id": row["record_id"],
                "image_path": row["image_path"],
                "tokens": [w["text"] for w in words],
                "bboxes": [w["bbox"] for w in words],
                "labels": [label2id.get(w.get("label", "O"), 0) for w in words],
                "is_noisy": row.get("is_noisy", False),
            }
        )
    out = out_dir or (Path(rendered_path).parent / "extraction_prepared")
    out.mkdir(parents=True, exist_ok=True)
    for split, items in buckets.items():
        (out / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + ("\n" if items else ""),
            encoding="utf-8",
        )
    write_json(out / "label2id.json", label2id)
    write_json(out / "summary.json", {k: len(v) for k, v in buckets.items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(prepare(args.inp, Config.load(), args.out))


if __name__ == "__main__":
    main()
