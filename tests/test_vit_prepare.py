"""Tests for ViT image-classification dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.classification.prepare_image_dataset import prepare
from src.utils.config import Config


def test_prepare_image_dataset(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    rendered = tmp_path / "rendered.jsonl"
    rows = []
    for i, label in enumerate(
        ["loss_notice", "application_commercial", "certificate_evidence", "loss_notice"]
    ):
        img_path = images / f"doc{i}.png"
        Image.new("RGB", (64, 64), color=(255, 255, 255)).save(img_path)
        rows.append(
            {
                "record_id": f"rec-{i}",
                "document_type": label,
                "image_path": str(img_path),
                "is_noisy": False,
            }
        )
    # missing image + unknown label should be skipped
    rows.append(
        {
            "record_id": "missing-img",
            "document_type": "loss_notice",
            "image_path": str(images / "nope.png"),
        }
    )
    rows.append(
        {
            "record_id": "bad-label",
            "document_type": "not_a_real_type",
            "image_path": str(images / "doc0.png"),
        }
    )
    rendered.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    splits = {
        "train": ["rec-0", "rec-1"],
        "val": ["rec-2"],
        "test": ["rec-3"],
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    monkeypatch.setenv("SPLITS_PATH", str(splits_path))

    cfg = Config.load()
    out = prepare(rendered, cfg, out_dir=tmp_path / "vit_prepared")
    assert (out / "train.jsonl").exists()
    assert (out / "label2id.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["train"] == 2
    assert summary["val"] == 1
    assert summary["test"] == 1
    assert summary["skipped"] == 2
    assert summary["modality"] == "image"

    train = [
        json.loads(line)
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {r["label"] for r in train} == {"loss_notice", "application_commercial"}
    assert all(Path(r["image_path"]).is_file() for r in train)
