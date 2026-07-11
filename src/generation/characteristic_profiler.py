"""
Build / refresh characteristic profiles from ingested public samples.

Committed profiles under data/profiles/ are the default. Running this module
merges any newly ingested raw samples into those profiles without removing
bundled priors.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.config import Config
from src.utils.io import read_json, write_json
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _load_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return default


def refresh_layout_profile(cfg: Config) -> dict[str, Any]:
    path = cfg.profiles_dir / "layout_profile.json"
    profile = _load_or_default(path, {"version": "1.0.0", "field_labels": []})
    funsd_path = cfg.raw_data_dir / "funsd_samples.json"
    if funsd_path.exists():
        funsd = read_json(funsd_path)
        labels = list(profile.get("field_labels") or [])
        for lab in funsd.get("field_labels") or []:
            if lab not in labels and len(lab) < 40:
                labels.append(lab)
        profile["field_labels"] = labels[:250]
        profile["source"] = "funsd_refresh_merged_with_bundled_priors"
    write_json(path, profile)
    return profile


def refresh_legal_style_profile(cfg: Config) -> dict[str, Any]:
    path = cfg.profiles_dir / "legal_style_profile.json"
    profile = _load_or_default(path, {"version": "1.0.0", "vocabulary_ngrams": [], "phrase_bank": []})
    legal_path = cfg.raw_data_dir / "legal_writing_samples.json"
    if not legal_path.exists():
        write_json(path, profile)
        return profile

    legal = read_json(legal_path)
    texts = [s.get("text", "") for s in legal.get("samples") or [] if s.get("text")]
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    for text in texts:
        words = [w.lower() for w in WORD_RE.findall(text)]
        unigrams.update(words)
        bigrams.update(f"{a} {b}" for a, b in zip(words, words[1:]))

    existing = list(profile.get("vocabulary_ngrams") or [])
    for ng, _ in bigrams.most_common(40):
        if ng not in existing:
            existing.append(ng)
    profile["vocabulary_ngrams"] = existing[:80]
    profile["source"] = "legal_writing_refresh_merged_with_bundled_priors"
    # Keep discourse markers / templates from bundled profile
    write_json(path, profile)
    return profile


def refresh_ocr_noise_profile(cfg: Config) -> dict[str, Any]:
    path = cfg.profiles_dir / "ocr_noise_profile.json"
    profile = _load_or_default(
        path,
        {
            "version": "1.0.0",
            "char_substitution_rate": 0.025,
            "char_deletion_rate": 0.008,
            "char_insertion_rate": 0.006,
        },
    )
    # FUNSD is already noisy scanned forms; keep rates stable unless we observe
    # extreme token lengths suggesting heavier noise.
    funsd_path = cfg.raw_data_dir / "funsd_samples.json"
    if funsd_path.exists():
        funsd = read_json(funsd_path)
        lengths = [s.get("n_words", 0) for s in funsd.get("samples") or []]
        if lengths:
            profile["observed_mean_words"] = sum(lengths) / len(lengths)
            profile["source"] = "funsd_noise_shape_priors"
    write_json(path, profile)
    return profile


def refresh_document_surface_profile(cfg: Config) -> dict[str, Any]:
    path = cfg.profiles_dir / "document_surface_profile.json"
    profile = _load_or_default(path, {"version": "1.0.0"})
    rvl_path = cfg.raw_data_dir / "rvl_cdip_samples.json"
    if rvl_path.exists():
        rvl = read_json(rvl_path)
        profile["observed_rvl_counts"] = rvl.get("counts") or {}
        profile["source"] = "rvl_cdip_style_mapping_refresh"
    write_json(path, profile)
    return profile


def ensure_insurance_distributions(cfg: Config) -> dict[str, Any]:
    path = cfg.profiles_dir / "insurance_distributions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing bundled insurance distributions at {path}")
    return read_json(path)


def run_profiler(cfg: Config) -> dict[str, str]:
    cfg.profiles_dir.mkdir(parents=True, exist_ok=True)
    ensure_insurance_distributions(cfg)
    refresh_layout_profile(cfg)
    refresh_legal_style_profile(cfg)
    refresh_ocr_noise_profile(cfg)
    refresh_document_surface_profile(cfg)
    versions = {
        p.name: read_json(p).get("version", "unknown")
        for p in sorted(cfg.profiles_dir.glob("*.json"))
    }
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id="characteristic-profiling-batch",
            stage="characteristic_profiling",
            source="data/profiles",
            prompt_version="profiles_v1",
            model=None,
            extra={"profile_versions": versions},
        ),
    )
    return versions


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Refresh characteristic profiles")
    parser.parse_args()
    cfg = Config.load()
    versions = run_profiler(cfg)
    print(json.dumps(versions, indent=2))


if __name__ == "__main__":
    main()
