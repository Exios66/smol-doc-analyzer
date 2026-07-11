from __future__ import annotations

from pathlib import Path

from src.generation.noise_injection import garble_text
from src.generation.skeleton_sampler import sample_batch, validate_skeleton
from src.generation.stage_a_document_gen import render_document_template
from src.generation.stage_b_memo_gen import render_memo_template
from src.utils.config import Config
from src.utils.io import read_json


def test_skeleton_sampling_validates():
    cfg = Config.load()
    dist = read_json(cfg.profiles_dir / "insurance_distributions.json")
    schema = read_json(cfg.claim_schema_path)
    skeletons = sample_batch(n=24, seed=0, dist=dist, schema=schema)
    assert len(skeletons) == 24
    types = {s["document_type"] for s in skeletons}
    assert len(types) >= 6
    for s in skeletons:
        validate_skeleton(s, schema)


def test_template_document_and_memo_and_noise():
    cfg = Config.load()
    dist = read_json(cfg.profiles_dir / "insurance_distributions.json")
    schema = read_json(cfg.claim_schema_path)
    layout = read_json(cfg.profiles_dir / "layout_profile.json")
    surface = read_json(cfg.profiles_dir / "document_surface_profile.json")
    legal = read_json(cfg.profiles_dir / "legal_style_profile.json")
    ocr = read_json(cfg.profiles_dir / "ocr_noise_profile.json")
    sk = sample_batch(n=1, seed=1, dist=dist, schema=schema)[0]
    import random

    rng = random.Random(1)
    text = render_document_template(sk, layout, surface, legal, rng)
    assert "Claim Number:" in text
    assert sk["claim_id"] in text
    doc = {"skeleton": sk, "document_type": sk["document_type"], "text": text, "claim_id": sk["claim_id"]}
    memo = render_memo_template(doc, legal, rng)
    assert "ADJUSTER MEMO" in memo
    noisy = garble_text(text, ocr, rng)
    assert isinstance(noisy, str)
    assert len(noisy) > 0
