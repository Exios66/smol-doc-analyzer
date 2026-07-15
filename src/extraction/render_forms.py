"""Render synthetic documents as page images with word bounding boxes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.utils.io import load_jsonl, write_jsonl


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def render_page(
    text: str, width: int = 1000, margin: int = 40
) -> tuple[Image.Image, list[dict[str, Any]], bool]:
    font = _font(16)
    # Estimate height from line count; grow canvas so long docs are not clipped.
    lines = text.split("\n")
    line_h = 22
    height = max(1200, margin * 2 + line_h * (len(lines) + 4))
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    words_meta: list[dict[str, Any]] = []
    y = margin
    truncated = False
    for line_idx, line in enumerate(lines):
        x = margin
        if not line.strip():
            y += line_h
            # Mark blank lines with a sentinel so BIO labeling can stop at EOL.
            words_meta.append(
                {
                    "text": "",
                    "bbox": [x, y, x, y + line_h],
                    "line_break": True,
                }
            )
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
                    "line_idx": line_idx,
                }
            )
            x = bbox[2] + 8
            if x > width - margin:
                x = margin
                y += line_h
        y += line_h
        words_meta.append(
            {
                "text": "",
                "bbox": [margin, y, margin, y + line_h],
                "line_break": True,
            }
        )
        if y > height - margin:
            truncated = True
            break
    # Drop trailing empty line-break sentinels from the labeling stream is fine;
    # keep them so label_words can detect end-of-line.
    return img, words_meta, truncated


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

_SECTION_HEADER = re.compile(r"^[A-Z][A-Z /&-]{2,}$")
_KEY_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9 /-]{1,40}:$")


def _is_section_header(text: str) -> bool:
    """True for ALL-CAPS section titles, not IDs/amounts that contain digits."""
    if any(ch.isdigit() for ch in text):
        return False
    return bool(_SECTION_HEADER.match(text))


def _starts_field_prefix(words: list[dict[str, Any]], i: int) -> bool:
    for _, prefix in FIELD_PATTERNS:
        pref_tokens = prefix.split(" ")
        window = [words[i + k]["text"] for k in range(len(pref_tokens)) if i + k < len(words)]
        if window == pref_tokens:
            return True
    return False


def label_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign BIO field labels; stop at EOL, next field, or section headers."""
    labeled: list[dict[str, Any]] = []
    i = 0
    while i < len(words):
        if words[i].get("line_break") or not str(words[i].get("text") or "").strip():
            # Skip blank / line-break sentinels in the gold stream.
            i += 1
            continue

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
            first = True
            while i < len(words):
                token = words[i]
                text = str(token.get("text") or "")
                if token.get("line_break") or not text.strip():
                    break
                if _starts_field_prefix(words, i):
                    break
                if _is_section_header(text) or _KEY_LIKE.match(text):
                    break
                # Stop before headers like "ACORD" that are not FIELD_PATTERNS values.
                if text.upper() in {"ACORD", "NARRATIVE"} and not first:
                    break
                tag = f"B-{field}" if first else f"I-{field}"
                labeled.append({**token, "label": tag})
                first = False
                i += 1
                # Safety cap for unusually long values on a single line.
                run = 0
                for item in reversed(labeled):
                    if item["label"].endswith(field) and item["label"] != "O":
                        run += 1
                    else:
                        break
                if run >= 24:
                    break
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
        img, words, truncated = render_page(doc["text"])
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
                "width": img.width,
                "height": img.height,
                "truncated": truncated,
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
