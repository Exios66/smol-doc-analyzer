"""Prepare classification datasets from Stage A synthetic documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json

logger = logging.getLogger(__name__)


def load_label_list(taxonomy_path: Path) -> list[str]:
    tax = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    # Training targets for Stage A docs exclude adjuster_memo (Stage B output)
    labels = [c["label"] for c in tax["categories"] if c["label"] != "adjuster_memo"]
    return labels


def assign_split(record_id: str, splits: dict[str, Any]) -> str:
    if record_id in splits.get("train", []):
        return "train"
    if record_id in splits.get("val", []):
        return "val"
    if record_id in splits.get("test", []):
        return "test"
    # Hash-stable stratified fallback (avoid silently dumping everything into train).
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def prepare(docs_path: Path, cfg: Config, out_dir: Path | None = None) -> Path:
    docs = load_jsonl(docs_path)
    labels = load_label_list(cfg.taxonomy_path)
    label2id = {l: i for i, l in enumerate(labels)}
    splits = read_json(cfg.splits_path) if cfg.splits_path.exists() else {"train": [], "val": [], "test": []}

    rows = {"train": [], "val": [], "test": []}
    miss = 0
    known = set(splits.get("train", []) + splits.get("val", []) + splits.get("test", []))
    for doc in docs:
        label = doc["document_type"]
        if label not in label2id:
            continue
        rid = doc["record_id"]
        if known and rid not in known:
            miss += 1
        split = assign_split(rid, splits)
        rows[split].append(
            {
                "record_id": rid,
                "text": doc["text"],
                "label": label,
                "label_id": label2id[label],
            }
        )

    if miss:
        rate = miss / max(len(docs), 1)
        logger.warning(
            "classification prepare: %s/%s record_ids missing from splits (%.0f%%); "
            "used hash-stable fallback",
            miss,
            len(docs),
            100 * rate,
        )
        if rate > 0.5 and known:
            raise ValueError(
                f"Too many unmatched record_ids ({miss}/{len(docs)}). "
                "Regenerate skeletons/splits or align Stage A `_record_id` with splits.json."
            )

    out = out_dir or (cfg.document_output_dir / "classification_prepared")
    out.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        (out / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + ("\n" if items else ""),
            encoding="utf-8",
        )
    write_json(out / "label2id.json", label2id)
    write_json(
        out / "summary.json",
        {k: len(v) for k, v in rows.items()} | {"labels": labels, "split_misses": miss},
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
