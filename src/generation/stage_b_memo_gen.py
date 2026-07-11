"""Stage B: generate adjuster-style memos from skeletons + Stage A documents."""

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
PROMPT_VERSION = "stage_b_v1"


def render_memo_template(
    doc: dict[str, Any],
    legal: dict[str, Any],
    rng: random.Random,
) -> str:
    sk = doc["skeleton"]
    policy = sk["policy"]
    loss = sk["loss_event"]
    parties = sk["parties"]
    fin = sk["financials"]
    templates = legal.get("reasoning_templates") or [
        "Issue: whether {issue}. Rule: coverage turns on {rule}. Application: {application}. Conclusion: {conclusion}."
    ]
    tpl = rng.choice(templates)
    reasoning = tpl.format(
        issue=f"coverage applies to a {loss['loss_type']} loss under {policy['coverage_type']}",
        rule="the policy declarations, conditions, and applicable exclusions",
        application=f"the described event ({loss['description_seed']}) at {loss['location']}",
        conclusion=(
            "proceed with investigation and reserve adequacy review"
            if sk["narrative_complexity"] != "fraud_flagged"
            else "escalate for SIU review while preserving a reservation of rights"
        ),
        duty="reasonable care consistent with policy conditions",
        breach=loss["description_seed"],
        causation=f"alleged {loss['loss_type']} on {loss['date_of_loss']}",
        damages=f"${fin['estimated_damage']:,.2f} subject to a ${fin['deductible']:,.2f} deductible",
        finding=f"the claim file for {sk['claim_id']} is {sk['narrative_complexity']}",
        because=loss["description_seed"],
        evidence="the Stage A source document and reported loss details",
    )
    phrase = rng.choice(legal.get("phrase_bank") or ["Based on the available evidence"])
    return "\n".join(
        [
            f"ADJUSTER MEMO — {sk['claim_id']}",
            f"To: Claims File",
            f"From: {parties['adjuster_assigned']}",
            f"Re: {policy['policyholder_name']} / {policy['policy_number']}",
            "",
            "Summary",
            f"{phrase} the reported {loss['loss_type']} loss on {loss['date_of_loss']} "
            f"under {policy['coverage_type']} in {policy['state']}.",
            "",
            "Facts",
            f"- Location: {loss['location']}",
            f"- Description: {loss['description_seed']}",
            f"- Police report: {'yes' if loss.get('police_report_filed') else 'no'}",
            f"- Injuries: {'yes' if loss.get('injuries_reported') else 'no'}",
            f"- Estimated damage: ${fin['estimated_damage']:,.2f}",
            f"- Deductible: ${fin['deductible']:,.2f}",
            f"- Current reserve: ${fin['reserve_set']:,.2f}",
            "",
            "Analysis",
            reasoning,
            "",
            "Next Steps",
            "- Confirm coverage grant/denial points in writing",
            "- Update reserve if investigation changes exposure",
            "- Request any missing supporting evidence",
            "",
            f"Source document type: {doc['document_type']}",
        ]
    )


def llm_memo(doc: dict[str, Any], cfg: Config, legal: dict[str, Any]) -> str:
    from src.utils.llm_client import GenerationClient

    client = GenerationClient(cfg)
    system = (
        "You write concise fictional insurance adjuster memos for ML training. "
        "Ground every factual claim in the skeleton. Use legal-adjacent reasoning style "
        "(issue/rule/application/conclusion) without producing a court opinion."
    )
    user = (
        f"Skeleton: {doc['skeleton']}\n"
        f"Source document excerpt:\n{doc['text'][:1500]}\n"
        f"Style vocabulary: {legal.get('vocabulary_ngrams', [])[:10]}\n"
        "Write the memo text only."
    )
    return client.generate(system, user, max_tokens=1000)


def generate_memos(docs: list[dict[str, Any]], cfg: Config, seed: int = 42) -> list[dict[str, Any]]:
    legal = read_json(cfg.profiles_dir / "legal_style_profile.json")
    rng = random.Random(seed)
    use_llm = bool(cfg.openrouter_api_key)
    memos: list[dict[str, Any]] = []
    for doc in docs:
        if use_llm:
            try:
                text = llm_memo(doc, cfg, legal)
                mode = "llm"
            except Exception as exc:
                logger.warning("LLM Stage B failed (%s); using template", exc)
                text = render_memo_template(doc, legal, rng)
                mode = "template_fallback"
        else:
            text = render_memo_template(doc, legal, rng)
            mode = "template"
        memos.append(
            {
                "record_id": doc["record_id"],
                "claim_id": doc["claim_id"],
                "document_type": doc["document_type"],
                "memo_text": text,
                "source_document_type": doc["document_type"],
                "generation_mode": mode,
                "prompt_version": PROMPT_VERSION,
            }
        )
    return memos


def run_stage_b(cfg: Config, inp: Path, out: Path | None = None, seed: int = 42) -> Path:
    docs = load_jsonl(inp)
    memos = generate_memos(docs, cfg, seed=seed)
    out_path = out or (cfg.memo_output_dir / f"memos_from_{inp.stem}.jsonl")
    write_jsonl(out_path, memos)
    log_provenance(
        cfg.provenance_log_path,
        ProvenanceRecord(
            record_id=f"stage-b-{inp.stem}",
            stage="stage_b_memo_gen",
            source=str(inp),
            prompt_version=PROMPT_VERSION,
            model=cfg.generation_model if cfg.openrouter_api_key else "template_renderer",
            extra={"n": len(memos), "out": str(out_path)},
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
    print(run_stage_b(cfg, args.inp, args.out, seed=args.seed))


if __name__ == "__main__":
    main()
