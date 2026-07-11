"""Prepare ViT image-classification datasets from rendered document pages.

Mirrors the Kaggle / Hugging Face ViT document-image workflow: each example is
an image path + taxonomy label (not OCR text).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.classification.prepare_dataset import assign_split, load_label_list
from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json


def prepare(rendered_path: Path, cfg: Config, out_dir: Path | None = None) -> Path:
    rows_in = load_jsonl(rendered_path)
    labels = load_label_list(cfg.taxonomy_path)
    label2id = {label: i for i, label in enumerate(labels)}
    splits = (
        read_json(cfg.splits_path)
        if cfg.splits_path.exists()
        else {"train": [], "val": [], "test": []}
    )

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    skipped = 0
    for row in rows_in:
        label = row.get("document_type")
        image_path = row.get("image_path")
        if label not in label2id or not image_path:
            skipped += 1
            continue
        path = Path(image_path)
        if not path.is_file():
            skipped += 1
            continue
        split = assign_split(row["record_id"], splits)
        rows[split].append(
            {
                "record_id": row["record_id"],
                "image_path": str(path),
                "label": label,
                "label_id": label2id[label],
                "is_noisy": bool(row.get("is_noisy", False)),
            }
        )

    out = out_dir or (Path(rendered_path).parent / "vit_classification_prepared")
    out.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        (out / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + ("\n" if items else ""),
            encoding="utf-8",
        )
    write_json(out / "label2id.json", label2id)
    write_json(
        out / "summary.json",
        {
            **{k: len(v) for k, v in rows.items()},
            "labels": labels,
            "skipped": skipped,
            "source": str(rendered_path),
            "modality": "image",
            "model_family": "vit",
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ViT document-image classification splits from rendered.jsonl"
    )
    parser.add_argument("--in", dest="inp", type=Path, required=True, help="rendered.jsonl path")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = Config.load()
    print(prepare(args.inp, cfg, args.out))


if __name__ == "__main__":
    main()
