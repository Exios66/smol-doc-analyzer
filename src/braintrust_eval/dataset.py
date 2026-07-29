"""Build the fixed-size (1024×1024) 10-per-class RVL-CDIP sample set.

Outputs land under ``data/braintrust/fixed_size_sampled/``:

- ``images/{label_id:02d}_{slug}/{document_id_safe}.png`` — padded PNGs
- ``samples.jsonl`` — manifest rows with local image paths + gold labels
- ``manifest.json`` — seed / counts / sizing metadata
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.braintrust_eval.classifier import to_underscore_label
from src.rvl_cdip.paths import LABEL_NAMES
from src.rvl_cdip.sample_images import materialize_samples
from src.rvl_cdip.store import RvlCdipStore
from src.utils.config import REPO_ROOT
from src.utils.io import write_json, write_jsonl

DEFAULT_SEED = 42
DEFAULT_N_PER_CLASS = 10
DEFAULT_SIZE = 1024
DEFAULT_SPLIT = "test"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "braintrust" / "fixed_size_sampled"


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _safe_doc_id(document_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", document_id)[:180]


def pad_to_square_png(
    src: Path,
    dest: Path,
    *,
    size: int = DEFAULT_SIZE,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> dict[str, int]:
    """Resize preserving aspect ratio and pad to ``size×size`` PNG."""
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(size / float(w), size / float(h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), fill)
        offset = ((size - nw) // 2, (size - nh) // 2)
        canvas.paste(resized, offset)
        canvas.save(dest, format="PNG", optimize=True)
    return {"src_w": w, "src_h": h, "out_w": size, "out_h": size}


def make_placeholder_png(dest: Path, *, size: int = DEFAULT_SIZE, label: str = "") -> None:
    """Write a solid placeholder PNG for dry / CI dataset builds."""
    from PIL import Image, ImageDraw

    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    margin = max(2, size // 16)
    draw.rectangle(
        [margin, margin, size - margin - 1, size - margin - 1],
        outline=(80, 80, 100),
        width=max(1, size // 64),
    )
    draw.text((margin + 2, size // 2), (label or dest.stem)[:40], fill=(40, 40, 60))
    img.save(dest, format="PNG", optimize=True)


def fetch_candidates(
    store: RvlCdipStore,
    label_id: int,
    *,
    split: str = DEFAULT_SPLIT,
) -> list[dict[str, Any]]:
    sql = """
        SELECT d.document_id, d.split, d.label_id, l.name AS label,
               d.image_relpath, d.image_abspath, d.source_line
        FROM documents d
        JOIN labels l ON l.label_id = d.label_id
        WHERE d.label_id = ? AND d.split = ?
    """
    return store.query(sql, (label_id, split), max_rows=50_000)


def sample_rows(
    store: RvlCdipStore,
    *,
    n_per_class: int = DEFAULT_N_PER_CLASS,
    seed: int = DEFAULT_SEED,
    split: str = DEFAULT_SPLIT,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label_id in range(len(LABEL_NAMES)):
        candidates = fetch_candidates(store, label_id, split=split)
        if len(candidates) < n_per_class:
            raise RuntimeError(
                f"label_id={label_id} ({LABEL_NAMES[label_id]!r}) has only "
                f"{len(candidates)} {split} candidates; need {n_per_class}. "
                "Run: python -m src.rvl_cdip build"
            )
        picked = rng.sample(candidates, n_per_class)
        for row in picked:
            item = dict(row)
            item["sample_seed"] = seed
            item["n_per_class"] = n_per_class
            item["split_filter"] = split
            out.append(item)
    return out


def build_fixed_size_sampled(
    *,
    out_dir: Path | None = None,
    n_per_class: int = DEFAULT_N_PER_CLASS,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_SIZE,
    split: str = DEFAULT_SPLIT,
    store: RvlCdipStore | None = None,
    materialize: bool = True,
    placeholder: bool = False,
    samples: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the fixed-size sample directory + JSONL manifest.

    ``placeholder=True`` writes synthetic PNGs (no RVL archive needed) — for
    CI / Braintrust wiring tests. Live experiments need the RVL index + images.
    """
    dest = Path(out_dir or DEFAULT_OUT_DIR)
    images_root = dest / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    if samples is None:
        if placeholder:
            rows: list[dict[str, Any]] = []
            for label_id, label in enumerate(LABEL_NAMES):
                for i in range(n_per_class):
                    rows.append(
                        {
                            "document_id": f"placeholder:{label_id:02d}:{i:02d}",
                            "split": split,
                            "label_id": label_id,
                            "label": label,
                            "image_relpath": f"placeholder/{label_id}/{i}.png",
                            "image_abspath": None,
                            "sample_seed": seed,
                            "n_per_class": n_per_class,
                            "split_filter": split,
                        }
                    )
            samples = rows
        else:
            db = store or RvlCdipStore()
            summary = db.summary()
            if int(summary.get("documents") or 0) == 0:
                raise RuntimeError(
                    "RVL-CDIP index is empty. Run: python -m src.rvl_cdip build"
                )
            samples = sample_rows(
                db, n_per_class=n_per_class, seed=seed, split=split
            )
            if materialize:
                mats = materialize_samples(list(samples), run_ocr=False)
                by_id = {m.document_id: m for m in mats}
                enriched: list[dict[str, Any]] = []
                for row in samples:
                    item = dict(row)
                    mat = by_id.get(str(item["document_id"]))
                    if mat and mat.image_path is not None:
                        item["image_abspath"] = str(mat.image_path)
                    elif mat and mat.error:
                        item["materialize_error"] = mat.error
                    enriched.append(item)
                samples = enriched

    exported: list[dict[str, Any]] = []
    for row in samples:
        label_id = int(row["label_id"])
        label = str(row["label"])
        doc_id = str(row["document_id"])
        rel_dir = f"{label_id:02d}_{_slug(label)}"
        out_png = images_root / rel_dir / f"{_safe_doc_id(doc_id)}.png"

        if placeholder or not row.get("image_abspath"):
            make_placeholder_png(out_png, size=size, label=label)
            geom = {"src_w": size, "src_h": size, "out_w": size, "out_h": size}
            used_placeholder = True
        else:
            src = Path(str(row["image_abspath"]))
            if not src.is_file():
                make_placeholder_png(out_png, size=size, label=label)
                geom = {"src_w": size, "src_h": size, "out_w": size, "out_h": size}
                used_placeholder = True
            else:
                geom = pad_to_square_png(src, out_png, size=size)
                used_placeholder = False

        label_underscore = to_underscore_label(label)
        exported.append(
            {
                "document_id": doc_id,
                "split": row.get("split"),
                "label_id": label_id,
                "label": label,
                "label_underscore": label_underscore,
                "image_relpath": row.get("image_relpath"),
                "source_image_abspath": row.get("image_abspath"),
                "fixed_image_path": str(out_png),
                "fixed_image_relpath": str(out_png.relative_to(dest)),
                "size": size,
                "placeholder": used_placeholder,
                "geometry": geom,
                "sample_seed": row.get("sample_seed", seed),
                "n_per_class": n_per_class,
                "split_filter": split,
            }
        )

    samples_path = dest / "samples.jsonl"
    manifest_path = dest / "manifest.json"
    write_jsonl(samples_path, exported)
    counts = Counter(int(r["label_id"]) for r in exported)
    manifest = {
        "experiment": "fixed_size_sampled",
        "seed": seed,
        "n_per_class": n_per_class,
        "n_classes": len(LABEL_NAMES),
        "total_samples": len(exported),
        "split_filter": split,
        "image_size": size,
        "label_names": list(LABEL_NAMES),
        "counts_by_label_id": {str(k): counts[k] for k in sorted(counts)},
        "placeholder": bool(placeholder),
        "samples_path": str(samples_path),
        "images_dir": str(images_root),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_samples(path: Path | None = None) -> list[dict[str, Any]]:
    samples_path = Path(path or (DEFAULT_OUT_DIR / "samples.jsonl"))
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"Missing sample manifest: {samples_path}. "
            "Run: python -m src.braintrust_eval build-dataset"
        )
    return [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
