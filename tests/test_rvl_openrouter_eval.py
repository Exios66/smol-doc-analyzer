"""Unit tests for RVL OpenRouter eval helpers (no live API calls)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from src.rvl_cdip import openrouter_eval as ore
from src.rvl_cdip import paths as rvl_paths
from src.rvl_cdip import sample_images as si
from src.utils.llm_client import encode_image_data_url


@pytest.fixture()
def venv_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_repo = tmp_path / "repo"
    fake_venv = fake_repo / ".venv"
    fake_venv.mkdir(parents=True)
    monkeypatch.setattr(rvl_paths, "REPO_ROOT", fake_repo)
    return fake_repo


def _tiny_png_bytes() -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color=(200, 220, 210)).save(buf, format="PNG")
    return buf.getvalue()


def test_encode_image_data_url_jpeg(tmp_path: Path):
    pytest.importorskip("PIL")
    path = tmp_path / "page.png"
    path.write_bytes(_tiny_png_bytes())
    url = encode_image_data_url(path, max_long_edge=64)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 40


def test_extract_image_from_archive(venv_layout: Path):
    pytest.importorskip("PIL")
    rel = "imagesa/a/b/c/abc00c00/000.tif"
    arch = rvl_paths.archive_path()
    arch.parent.mkdir(parents=True, exist_ok=True)
    png = _tiny_png_bytes()
    # Store as TIFF-named PNG bytes — Pillow still opens; extract only cares about bytes.
    with tarfile.open(arch, "w:gz") as tf:
        info = tarfile.TarInfo(name=rel)
        info.size = len(png)
        tf.addfile(info, io.BytesIO(png))

    out = si.extract_image_from_archive(rel)
    assert out is not None
    assert out.is_file()
    assert out.read_bytes() == png

    # Second call hits cache on disk
    again = si.extract_image_from_archive(rel)
    assert again == out


def test_materialize_prefers_existing_abspath(venv_layout: Path, tmp_path: Path):
    img = rvl_paths.images_dir() / "x" / "doc.tif"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"abc")
    sample = {
        "document_id": "train:x/doc.tif",
        "image_relpath": "x/doc.tif",
        "image_abspath": str(img),
    }
    result = si.materialize_sample_image(sample)
    assert result.image_path == img.resolve()
    assert result.extracted is False
    assert result.error is None


def test_normalize_rvl_label():
    assert ore.normalize_rvl_label("Invoice") == "invoice"
    assert ore.normalize_rvl_label("scientific_report") == "scientific report"
    assert ore.normalize_rvl_label("scientific report") == "scientific report"


def test_parse_json_object_fenced():
    raw = 'Here you go:\n```json\n{"summary": "ok", "parties": []}\n```'
    obj = ore.parse_json_object(raw)
    assert obj.get("_parse_error") is not True
    assert obj["summary"] == "ok"


def test_load_samples_max_per_class(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    rows = []
    for label_id in range(3):
        for i in range(5):
            rows.append(
                {
                    "document_id": f"train:c{label_id}-{i}.tif",
                    "split": "train",
                    "label_id": label_id,
                    "label": ore.RVL_LABELS[label_id],
                    "image_relpath": f"c{label_id}-{i}.tif",
                    "image_abspath": None,
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    from collections import Counter

    loaded = ore.load_samples(path, max_per_class=2)
    assert len(loaded) == 6
    counts = Counter(int(r["label_id"]) for r in loaded)
    assert counts == {0: 2, 1: 2, 2: 2}


def test_run_eval_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    samples_path = tmp_path / "samples.jsonl"
    samples = [
        {
            "document_id": "train:a.tif",
            "split": "train",
            "label_id": 0,
            "label": "letter",
            "image_relpath": "a.tif",
            "image_abspath": None,
        },
        {
            "document_id": "train:b.tif",
            "split": "train",
            "label_id": 11,
            "label": "invoice",
            "image_relpath": "b.tif",
            "image_abspath": None,
        },
    ]
    samples_path.write_text(
        "\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8"
    )
    out = tmp_path / "out"
    # Avoid materialize trying real archive; run vision-only dry-run with empty mats
    mats = [
        si.MaterializeResult(
            document_id=s["document_id"],
            image_relpath=s["image_relpath"],
            image_path=None,
            error="no image in test",
        )
        for s in samples
    ]
    # Vision without images records errors; use text modality with fake OCR
    mats = [
        si.MaterializeResult(
            document_id=s["document_id"],
            image_relpath=s["image_relpath"],
            image_path=tmp_path / f"{s['document_id'].replace(':', '_')}.tif",
            ocr_text="Claim Number: CLM-1\nInvoice total $12.00",
        )
        for s in samples
    ]
    for m in mats:
        assert m.image_path is not None
        m.image_path.write_bytes(b"x")

    monkeypatch.setattr(ore, "DEFAULT_SAMPLES_PATH", samples_path)
    manifest = ore.run_eval(
        samples,
        materialize=mats,
        vision_models=("openai/gpt-4o-mini",),
        text_models=("openai/gpt-4o-mini",),
        modalities=("vision", "text"),
        out_dir=out,
        dry_run=True,
        resume=False,
    )
    assert manifest["dry_run"] is True
    assert (out / "predictions.jsonl").is_file()
    assert (out / "summary_classification.json").is_file()
    assert (out / "summary_extraction.json").is_file()
    preds = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # 2 docs × 2 modalities × 1 model × 2 tasks
    assert len(preds) == 8
    cls = [p for p in preds if p["task"] == "classify" and not p.get("error")]
    assert cls
    assert all(p["prediction"] in ore.RVL_LABELS for p in cls)


def test_score_classification_accuracy():
    rows = [
        {
            "task": "classify",
            "model_id": "m",
            "modality": "vision",
            "label": "invoice",
            "prediction": "invoice",
            "error": None,
            "latency_seconds": 0.1,
            "input_tokens": 1,
            "output_tokens": 1,
        },
        {
            "task": "classify",
            "model_id": "m",
            "modality": "vision",
            "label": "memo",
            "prediction": "letter",
            "error": None,
            "latency_seconds": 0.1,
            "input_tokens": 1,
            "output_tokens": 1,
        },
    ]
    summary = ore.score_classification(rows)
    assert summary["per_model"][0]["accuracy"] == 0.5
