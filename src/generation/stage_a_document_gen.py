"""Stage A: generate synthetic insurance document text from claim skeletons."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_jsonl
from src.utils.provenance import ProvenanceRecord, log_provenance

logger = logging.getLogger(__name__)

PROMPT_VERSION = "stage_a_v1"


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _legal_flavor(rng: random.Random, legal: dict[str, Any], complexity: str) -> str:
    markers = legal.get("discourse_markers") or []
    phrases = legal.get("phrase_bank") or []
    vocab = legal.get("vocabulary_ngrams") or []
    bits = []
    if phrases:
        bits.append(rng.choice(phrases))
    if markers and complexity in {"ambiguous", "fraud_flagged"}:
        bits.append(rng.choice(markers).capitalize())
    if vocab:
        bits.append(f"Reference is made to {rng.choice(vocab)}.")
    return " ".join(bits)


def render_document_template(
    skeleton: dict[str, Any],
    layout: dict[str, Any],
    surface: dict[str, Any],
    legal: dict[str, Any],
    rng: random.Random,
) -> str:
    doc_type = skeleton["document_type"]
    openings = (surface.get("opening_cues") or {}).get(doc_type) or [doc_type.upper()]
    closings = (surface.get("closing_cues") or {}).get(doc_type) or ["Signature"]
    headers = (layout.get("section_headers_by_document_type") or {}).get(doc_type) or ["Details"]
    policy = skeleton["policy"]
    loss = skeleton["loss_event"]
    parties = skeleton["parties"]
    fin = skeleton["financials"]

    lines = [
        rng.choice(openings),
        f"Claim Number: {skeleton['claim_id']}",
        f"ACORD Form: {skeleton.get('acord_form_number') or 'N/A'}",
        "",
    ]
    for h in headers:
        lines.append(h.upper())
        if h.lower().startswith("applicant") or h.lower() in {"insured", "named insured"}:
            lines.append(f"Named Insured: {policy['policyholder_name']}")
            lines.append(f"Policy Number: {policy['policy_number']}")
            lines.append(f"State: {policy['state']}")
            lines.append(f"Coverage Type: {policy['coverage_type']}")
            lines.append(f"Effective Date: {policy['effective_date']}")
        elif "loss" in h.lower() or h.lower() == "description":
            lines.append(f"Date of Loss: {loss['date_of_loss']}")
            lines.append(f"Loss Type: {loss['loss_type']}")
            lines.append(f"Loss Location: {loss['location']}")
            lines.append(f"Description of Loss: {loss['description_seed']}")
            lines.append(f"Police Report Filed: {'Yes' if loss.get('police_report_filed') else 'No'}")
            lines.append(f"Injuries Reported: {'Yes' if loss.get('injuries_reported') else 'No'}")
        elif "coverage" in h.lower() or "limit" in h.lower() or "financial" in h.lower() or h == "Total":
            lines.append(f"Estimated Damage: {_money(fin['estimated_damage'])}")
            lines.append(f"Deductible: {_money(fin['deductible'])}")
            lines.append(f"Reserve Amount: {_money(fin['reserve_set'])}")
        elif "claim" in h.lower() or h.startswith("Re"):
            lines.append(f"Adjuster Name: {parties['adjuster_assigned']}")
            if parties.get("claimant"):
                lines.append(f"Claimant Name: {parties['claimant']}")
            lines.append(f"Claim Status: under review ({skeleton['narrative_complexity']})")
        elif "line" in h.lower() or "labor" in h.lower() or "parts" in h.lower():
            lines.append(f"Labor: {_money(fin['estimated_damage'] * 0.45)}")
            lines.append(f"Parts / Materials: {_money(fin['estimated_damage'] * 0.55)}")
            lines.append(f"Estimate Total: {_money(fin['estimated_damage'])}")
        else:
            lines.append(f"Policy Number: {policy['policy_number']}")
            lines.append(f"Named Insured: {parties['insured']}")
        lines.append("")

    if doc_type in (legal.get("apply_to_document_types") or []):
        lines.append("NARRATIVE")
        lines.append(
            f"{_legal_flavor(rng, legal, skeleton['narrative_complexity'])} "
            f"The loss is described as: {loss['description_seed']}. "
            f"Complexity assessment: {skeleton['narrative_complexity']}."
        )
        lines.append("")

    lines.append(rng.choice(closings))
    lines.append(f"Prepared for claim {skeleton['claim_id']}")
    return "\n".join(lines)


def llm_document(skeleton: dict[str, Any], cfg: Config, legal: dict[str, Any]) -> str:
    from src.utils.llm_client import GenerationClient

    client = GenerationClient(cfg)
    system = (
        "You generate fictional insurance operations documents for ML training. "
        "Use only the provided skeleton facts. Do not invent real people or real policies. "
        "When narrative is needed, you may use formal legal-adjacent vocabulary and reasoning "
        "style, but the document must remain an insurance form/correspondence, not a court filing."
    )
    user = (
        f"Document type: {skeleton['document_type']}\n"
        f"Skeleton JSON:\n{skeleton}\n"
        f"Optional style n-grams: {legal.get('vocabulary_ngrams', [])[:12]}\n"
        "Write the full document text only."
    )
    return client.generate(system, user, max_tokens=1200)


def generate_documents(
    skeletons: list[dict[str, Any]],
    cfg: Config,
    use_llm: bool,
    seed: int = 42,
) -> list[dict[str, Any]]:
    layout = read_json(cfg.profiles_dir / "layout_profile.json")
    surface = read_json(cfg.profiles_dir / "document_surface_profile.json")
    legal = read_json(cfg.profiles_dir / "legal_style_profile.json")
    rng = random.Random(seed)
    docs: list[dict[str, Any]] = []
    for sk in skeletons:
        if use_llm and cfg.openrouter_api_key:
            try:
                text = llm_document(sk, cfg, legal)
                mode = "llm"
            except Exception as exc:
                logger.warning("LLM Stage A failed (%s); using template", exc)
                text = render_document_template(sk, layout, surface, legal, rng)
                mode = "template_fallback"
        else:
            text = render_document_template(sk, layout, surface, legal, rng)
            mode = "template"
        docs.append(
            {
                "record_id": sk.get("_record_id")
                or f"{sk['claim_id']}::{sk['document_type']}",
                "claim_id": sk["claim_id"],
                "document_type": sk["document_type"],
                "text": text,
                "skeleton": sk,
                "generation_mode": mode,
                "prompt_version": PROMPT_VERSION,
            }
        )
    return docs


def run_stage_a(cfg: Config, inp: Path, out: Path | None = None, seed: int = 42) -> Path:
    skeletons = load_jsonl(inp)
    use_llm = bool(cfg.openrouter_api_key)
    docs = generate_documents(skeletons, cfg, use_llm=use_llm, seed=seed)
    out_path = out or (cfg.document_output_dir / f"documents_from_{inp.stem}.jsonl")
    write_jsonl(out_path, docs)
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"stage-a-{inp.stem}",
            stage="stage_a_document_gen",
            source=str(inp),
            prompt_version=PROMPT_VERSION,
            model=cfg.generation_model if use_llm else "template_renderer",
            extra={"n": len(docs), "out": str(out_path)},
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
    print(run_stage_a(cfg, args.inp, args.out, seed=args.seed))


if __name__ == "__main__":
    main()
