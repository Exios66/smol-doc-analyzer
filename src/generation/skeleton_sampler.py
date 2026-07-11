"""Randomized claim skeleton sampler seeded by insurance distribution profiles."""

from __future__ import annotations

import argparse
import json
import logging
import random
import string
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from src.utils.config import Config
from src.utils.io import read_json, write_json, write_jsonl
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)

FIRST_NAMES = [
    "Avery", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Cameron",
    "Harper", "Reese", "Skyler", "Drew", "Jamie", "Alex", "Parker", "Rowan",
]
LAST_NAMES = [
    "Nguyen", "Patel", "Garcia", "Johnson", "Kim", "Brown", "Martinez", "Lee",
    "Anderson", "Thompson", "Rivera", "Clark", "Lewis", "Walker", "Young", "Hall",
]
STREETS = [
    "Oak St", "Maple Ave", "Cedar Rd", "Pine Blvd", "Lakeview Dr", "Hillcrest Ln",
    "River Rd", "Sunset Ave", "Industrial Pkwy", "Market St",
]


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = [float(weights[k]) for k in keys]
    return rng.choices(keys, weights=vals, k=1)[0]


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _policy_number(rng: random.Random, state: str) -> str:
    return f"{state}-{rng.randint(100000, 999999)}-{rng.choice(string.ascii_uppercase)}"


def _claim_id(rng: random.Random, year: int) -> str:
    return f"CLM-{year}-{rng.randint(1, 999999):06d}"


def _lognormal_amount(rng: random.Random, spec: dict[str, Any]) -> float:
    val = rng.lognormvariate(float(spec["mu"]), float(spec["sigma"]))
    return round(min(max(val, float(spec["clip_min"])), float(spec["clip_max"])), 2)


def _random_date(rng: random.Random, start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=rng.randint(0, max(delta, 1)))).isoformat()


def sample_skeleton(
    rng: random.Random,
    dist: dict[str, Any],
    document_type: str | None = None,
    multi_doc_group_id: str | None = None,
) -> dict[str, Any]:
    doc_type = document_type or _weighted_choice(rng, dist["document_type_weights"])
    state = _weighted_choice(rng, dist["state_weights"])
    coverage = _weighted_choice(rng, dist["coverage_type_weights"])
    loss_type = _weighted_choice(rng, dist["loss_type_weights"])
    complexity = _weighted_choice(rng, dist["narrative_complexity_weights"])

    year = rng.randint(2023, 2026)
    loss_date = _random_date(rng, date(year, 1, 1), date(year, 12, 28))
    effective = _random_date(rng, date(year - 1, 1, 1), date(year, 6, 1))
    estimated = _lognormal_amount(rng, dist["estimated_damage"])
    deductible = float(
        rng.choices(dist["deductible_choices"], weights=dist["deductible_weights"], k=1)[0]
    )
    ratio_spec = dist["reserve_ratio"]
    ratio = min(
        max(rng.gauss(ratio_spec["mean"], ratio_spec["std"]), ratio_spec["clip_min"]),
        ratio_spec["clip_max"],
    )
    reserve = round(estimated * ratio, 2)

    seeds = dist["description_seeds"].get(loss_type) or dist["description_seeds"]["other"]
    acord_options = dist["acord_form_by_document_type"].get(doc_type, [None])
    acord = rng.choice(acord_options)

    insured = _person(rng)
    skeleton = {
        "claim_id": _claim_id(rng, year),
        "document_type": doc_type,
        "acord_form_number": acord,
        "policy": {
            "policy_number": _policy_number(rng, state),
            "policyholder_name": insured,
            "state": state,
            "coverage_type": coverage,
            "effective_date": effective,
        },
        "loss_event": {
            "date_of_loss": loss_date,
            "loss_type": loss_type,
            "location": f"{rng.randint(100, 9999)} {rng.choice(STREETS)}, {state}",
            "description_seed": rng.choice(seeds),
            "police_report_filed": rng.random() < float(dist["police_report_rate"]),
            "injuries_reported": rng.random() < float(dist["injuries_reported_rate"]),
        },
        "parties": {
            "insured": insured,
            "claimant": _person(rng) if rng.random() < 0.55 else None,
            "adjuster_assigned": _person(rng),
        },
        "financials": {
            "estimated_damage": estimated,
            "deductible": deductible,
            "reserve_set": reserve,
        },
        "narrative_complexity": complexity,
        "multi_doc_group_id": multi_doc_group_id,
        "target_outputs": {"document_text": None, "memo_text": None},
    }
    return skeleton


def validate_skeleton(skeleton: dict[str, Any], schema: dict[str, Any]) -> None:
    jsonschema.validate(instance=skeleton, schema=schema)


def sample_batch(
    n: int,
    seed: int,
    dist: dict[str, Any],
    schema: dict[str, Any],
    multi_doc_rate: float = 0.15,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    doc_types = list(dist["document_type_weights"].keys())

    # Ensure coverage across types first
    base = max(1, n // len(doc_types))
    planned: list[str | None] = []
    for dt in doc_types:
        planned.extend([dt] * base)
    while len(planned) < n:
        planned.append(None)
    planned = planned[:n]
    rng.shuffle(planned)

    i = 0
    while i < n:
        if i < n - 2 and rng.random() < multi_doc_rate:
            group_id = f"GRP-{rng.randint(10000, 99999)}"
            bundle_types = rng.sample(
                ["loss_notice", "repair_estimate", "supporting_evidence", "claims_correspondence"],
                k=min(3, n - i),
            )
            for bt in bundle_types:
                sk = sample_skeleton(rng, dist, document_type=bt, multi_doc_group_id=group_id)
                # share claim_id within bundle
                if out and out[-1].get("multi_doc_group_id") == group_id:
                    sk["claim_id"] = out[-1]["claim_id"]
                    sk["policy"] = dict(out[-1]["policy"])
                    sk["loss_event"] = dict(out[-1]["loss_event"])
                    sk["parties"]["insured"] = out[-1]["parties"]["insured"]
                    sk["parties"]["adjuster_assigned"] = out[-1]["parties"]["adjuster_assigned"]
                validate_skeleton(sk, schema)
                out.append(sk)
                i += 1
                if i >= n:
                    break
        else:
            sk = sample_skeleton(rng, dist, document_type=planned[i])
            validate_skeleton(sk, schema)
            out.append(sk)
            i += 1
    return out


def write_splits(skeletons: list[dict[str, Any]], splits_path: Path, seed: int = 42) -> dict[str, list[str]]:
    ids = [s["claim_id"] + "::" + s["document_type"] + f"::{idx}" for idx, s in enumerate(skeletons)]
    rng = random.Random(seed)
    order = list(range(len(ids)))
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, int(0.15 * n))
    n_val = max(1, int(0.15 * n))
    test_idx = order[:n_test]
    val_idx = order[n_test : n_test + n_val]
    train_idx = order[n_test + n_val :]
    splits = {
        "seed": seed,
        "train": [ids[i] for i in train_idx],
        "val": [ids[i] for i in val_idx],
        "test": [ids[i] for i in test_idx],
        "record_ids": ids,
    }
    write_json(splits_path, splits)
    return splits


def run_sampler(cfg: Config, n: int, seed: int, out: Path | None = None) -> Path:
    dist = read_json(cfg.profiles_dir / "insurance_distributions.json")
    schema = read_json(cfg.claim_schema_path)
    skeletons = sample_batch(n=n, seed=seed, dist=dist, schema=schema)
    out_path = out or (cfg.skeleton_output_dir / f"skeletons_n{n}_seed{seed}.jsonl")
    write_jsonl(out_path, skeletons)
    # annotate with split ids
    for idx, sk in enumerate(skeletons):
        sk["_record_id"] = f"{sk['claim_id']}::{sk['document_type']}::{idx}"
    write_jsonl(out_path, skeletons)
    write_splits(skeletons, cfg.splits_path, seed=seed)
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"skeletons-{seed}-{n}",
            stage="skeleton_sampling",
            source="insurance_distributions.json",
            prompt_version="skeleton_sampler_v1",
            model=None,
            extra={"n": n, "seed": seed, "out": str(out_path)},
        ),
    )
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = Config.load()
    path = run_sampler(cfg, n=args.n, seed=args.seed, out=args.out)
    print(path)


if __name__ == "__main__":
    main()
