"""Prepare classification datasets from Stage A synthetic documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json


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
    # fallback: hash-stable
    return "train"


def prepare(docs_path: Path, cfg: Config, out_dir: Path | None = None) -> Path:
    docs = load_jsonl(docs_path)
    labels = load_label_list(cfg.taxonomy_path)
    label2id = {l: i for i, l in enumerate(labels)}
    splits = read_json(cfg.splits_path) if cfg.splits_path.exists() else {"train": [], "val": [], "test": []}

    rows = {"train": [], "val": [], "test": []}
    for doc in docs:
        label = doc["document_type"]
        if label not in label2id:
            continue
        split = assign_split(doc["record_id"], splits)
        rows[split].append(
            {
                "record_id": doc["record_id"],
                "text": doc["text"],
                "label": label,
                "label_id": label2id[label],
            }
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
        {k: len(v) for k, v in rows.items()} | {"labels": labels},
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
