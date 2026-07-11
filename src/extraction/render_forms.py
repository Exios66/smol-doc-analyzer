"""Render synthetic documents as page images with word bounding boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.utils.config import Config
from src.utils.io import load_jsonl, write_jsonl


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def render_page(text: str, width: int = 1000, margin: int = 40) -> tuple[Image.Image, list[dict[str, Any]]]:
    font = _font(16)
    # Estimate height
    lines = text.split("\n")
    line_h = 22
    height = max(1200, margin * 2 + line_h * (len(lines) + 2))
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    words_meta: list[dict[str, Any]] = []
    y = margin
    for line in lines:
        x = margin
        if not line.strip():
            y += line_h
            continue
        for word in line.split(" "):
            if not word:
                x += 6
                continue
            bbox = draw.textbbox((x, y), word, font=font)
            draw.text((x, y), word, fill=(0, 0, 0), font=font)
            words_meta.append(
                {
                    "text": word,
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                }
            )
            x = bbox[2] + 8
            if x > width - margin:
                x = margin
                y += line_h
        y += line_h
        if y > height - margin:
            break
    return img, words_meta


FIELD_PATTERNS = [
    ("claim_id", "Claim Number:"),
    ("policy_number", "Policy Number:"),
    ("policyholder_name", "Named Insured:"),
    ("date_of_loss", "Date of Loss:"),
    ("loss_type", "Loss Type:"),
    ("location", "Loss Location:"),
    ("estimated_damage", "Estimated Damage:"),
    ("deductible", "Deductible:"),
    ("reserve_set", "Reserve Amount:"),
    ("adjuster_assigned", "Adjuster Name:"),
    ("claimant", "Claimant Name:"),
    ("effective_date", "Effective Date:"),
    ("state", "State:"),
    ("coverage_type", "Coverage Type:"),
]


def label_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign BIO-ish field labels by scanning for known prefixes on the page."""
    labeled = []
    active_field = "O"
    remaining_value_tokens = 0
    # Rebuild line-ish stream
    i = 0
    while i < len(words):
        w = words[i]["text"]
        # Look ahead for prefix patterns spanning multiple tokens
        matched = None
        for field, prefix in FIELD_PATTERNS:
            pref_tokens = prefix.split(" ")
            window = [words[i + k]["text"] for k in range(len(pref_tokens)) if i + k < len(words)]
            if window == pref_tokens:
                matched = (field, len(pref_tokens))
                break
        if matched:
            field, n = matched
            for k in range(n):
                labeled.append({**words[i + k], "label": "O"})
            i += n
            # following tokens until next known label-ish colon pattern get the field
            active_field = field
            # consume value tokens until end of "line" heuristic: next capitalized key or blank jump
            while i < len(words):
                # stop if next token starts a new field prefix
                stop = False
                for _, prefix in FIELD_PATTERNS:
                    pref_tokens = prefix.split(" ")
                    window = [words[i + k]["text"] for k in range(len(pref_tokens)) if i + k < len(words)]
                    if window == pref_tokens:
                        stop = True
                        break
                if stop:
                    break
                tag = f"B-{active_field}" if remaining_value_tokens == 0 else f"I-{active_field}"
                # use a local counter via whether previous was same field
                if labeled and labeled[-1]["label"].endswith(active_field) and labeled[-1]["label"] != "O":
                    tag = f"I-{active_field}"
                else:
                    tag = f"B-{active_field}"
                labeled.append({**words[i], "label": tag})
                i += 1
                # crude: stop after 12 tokens of value
                run = 0
                for item in reversed(labeled):
                    if item["label"].endswith(active_field):
                        run += 1
                    else:
                        break
                if run >= 12:
                    break
            active_field = "O"
            continue
        labeled.append({**words[i], "label": "O"})
        i += 1
    return labeled


def render_documents(docs_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    rows = []
    for doc in load_jsonl(docs_path):
        img, words = render_page(doc["text"])
        labeled = label_words(words)
        img_path = images_dir / f"{doc['record_id'].replace('::', '__')}.png"
        img.save(img_path)
        rows.append(
            {
                "record_id": doc["record_id"],
                "claim_id": doc["claim_id"],
                "document_type": doc["document_type"],
                "image_path": str(img_path),
                "words": labeled,
                "skeleton": doc.get("skeleton"),
                "is_noisy": bool(doc.get("is_noisy", False)),
            }
        )
    out_path = out_dir / "rendered.jsonl"
    write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(render_documents(args.inp, args.out))


if __name__ == "__main__":
    main()
