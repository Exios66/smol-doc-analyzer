"""OCR noise injection and multi-document bundle indexing."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json, write_jsonl
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)


def _substitute_char(ch: str, profile: dict[str, Any], rng: random.Random) -> str:
    digit_map = profile.get("digit_confusion") or {}
    if ch in digit_map and rng.random() < 0.5:
        return rng.choice(digit_map[ch])
    if ch.isalpha() and rng.random() < 0.3:
        artifacts = profile.get("common_ocr_artifacts") or ["|"]
        return rng.choice(artifacts)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return rng.choice(alphabet)


def garble_text(text: str, profile: dict[str, Any], rng: random.Random) -> str:
    sub_rate = float(profile.get("char_substitution_rate", 0.02))
    del_rate = float(profile.get("char_deletion_rate", 0.005))
    ins_rate = float(profile.get("char_insertion_rate", 0.005))
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        # multi-char letter confusions
        applied = False
        for src, alts in (profile.get("letter_confusion") or {}).items():
            if text.startswith(src, i) and rng.random() < sub_rate * 3:
                out.append(rng.choice(alts))
                i += len(src)
                applied = True
                break
        if applied:
            continue
        r = rng.random()
        if r < del_rate and ch not in "\n":
            i += 1
            continue
        if r < del_rate + ins_rate:
            out.append(rng.choice((profile.get("common_ocr_artifacts") or ["~"]) + [ch]))
        if r < del_rate + ins_rate + sub_rate and ch.isalnum():
            out.append(_substitute_char(ch, profile, rng))
        else:
            out.append(ch)
        if ch == "\n" and rng.random() < float(profile.get("line_break_noise_rate", 0.0)):
            out.append(" ")
        i += 1

    # word split / merge on a line basis
    lines = "".join(out).split("\n")
    noisy_lines = []
    for line in lines:
        words = line.split(" ")
        j = 0
        rebuilt: list[str] = []
        while j < len(words):
            w = words[j]
            if (
                j + 1 < len(words)
                and words[j + 1]
                and rng.random() < float(profile.get("word_merge_rate", 0.0))
            ):
                rebuilt.append(w + words[j + 1])
                j += 2
                continue
            if len(w) > 4 and rng.random() < float(profile.get("word_split_rate", 0.0)):
                cut = rng.randint(1, len(w) - 1)
                rebuilt.extend([w[:cut], w[cut:]])
            else:
                rebuilt.append(w)
            j += 1
        noisy_lines.append(" ".join(rebuilt))
    return "\n".join(noisy_lines)


def inject_noise(docs: list[dict[str, Any]], profile: dict[str, Any], seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    noisy = []
    for doc in docs:
        noisy.append(
            {
                **doc,
                "text": garble_text(doc["text"], profile, rng),
                "is_noisy": True,
                "noise_profile_version": profile.get("version"),
            }
        )
    return noisy


def build_multi_doc_index(docs: list[dict[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for doc in docs:
        sk = doc.get("skeleton") or {}
        group = sk.get("multi_doc_group_id")
        if group:
            index.setdefault(group, []).append(doc["record_id"])
    return index


def run_noise(cfg: Config, inp: Path, out: Path | None = None, seed: int = 42) -> Path:
    docs = load_jsonl(inp)
    profile = read_json(cfg.profiles_dir / "ocr_noise_profile.json")
    noisy = inject_noise(docs, profile, seed=seed)
    out_dir = cfg.noisy_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out or (out_dir / f"noisy_from_{inp.stem}.jsonl")
    write_jsonl(out_path, noisy)
    index = build_multi_doc_index(docs)
    write_json(out_dir / f"multi_doc_index_from_{inp.stem}.json", index)
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"noise-{inp.stem}",
            stage="noise_injection",
            source=str(inp),
            prompt_version="noise_injection_v1",
            model=None,
            extra={"n": len(noisy), "multi_doc_groups": len(index), "out": str(out_path)},
        ),
    )
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = Config.load()
    print(run_noise(cfg, args.inp, args.out, seed=args.seed))


if __name__ == "__main__":
    main()
