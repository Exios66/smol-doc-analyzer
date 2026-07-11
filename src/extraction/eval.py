"""Field-level extraction evaluation + failure mode notes."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json


def _decode_entities(tokens: list[str], labels: list[str]) -> dict[str, list[str]]:
    ents: dict[str, list[str]] = defaultdict(list)
    cur_field = None
    cur_tokens: list[str] = []
    for tok, lab in zip(tokens, labels):
        if lab == "O":
            if cur_field and cur_tokens:
                ents[cur_field].append(" ".join(cur_tokens))
            cur_field, cur_tokens = None, []
            continue
        prefix, _, field = lab.partition("-")
        if prefix == "B" or field != cur_field:
            if cur_field and cur_tokens:
                ents[cur_field].append(" ".join(cur_tokens))
            cur_field = field
            cur_tokens = [tok]
        else:
            cur_tokens.append(tok)
    if cur_field and cur_tokens:
        ents[cur_field].append(" ".join(cur_tokens))
    return ents


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


def _partial_match(a: str, b: str) -> bool:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def evaluate(
    model_dir: Path,
    prepared_dir: Path,
    cfg: Config,
    split: str = "test",
    noisy_prepared: Path | None = None,
) -> dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    rows = load_jsonl(prepared_dir / f"{split}.jsonl")
    label2id = {k: int(v) for k, v in read_json(model_dir / "label2id.json").items()}
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    token_true, token_pred = [], []
    exact_hits = defaultdict(int)
    partial_hits = defaultdict(int)
    totals = defaultdict(int)
    hard_fields = ["date_of_loss", "estimated_damage", "deductible", "reserve_set", "location"]

    with torch.no_grad():
        for row in rows:
            enc = tokenizer(
                row["tokens"],
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            word_ids = enc.word_ids(batch_index=0)
            logits = model(**enc).logits[0]
            pred_ids = torch.argmax(logits, dim=-1).tolist()
            aligned_pred = ["O"] * len(row["tokens"])
            seen = set()
            for idx, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                aligned_pred[wid] = id2label.get(int(pred_ids[idx]), "O")
            gold = [id2label.get(int(i), "O") for i in row["labels"]]
            for g, p in zip(gold, aligned_pred):
                token_true.append(label2id.get(g, 0))
                token_pred.append(label2id.get(p, 0))

            gold_ents = _decode_entities(row["tokens"], gold)
            pred_ents = _decode_entities(row["tokens"], aligned_pred)
            fields = set(gold_ents) | set(pred_ents)
            for field in fields:
                gvals = gold_ents.get(field, [])
                pvals = pred_ents.get(field, [])
                totals[field] += max(len(gvals), 1)
                # exact
                if gvals and pvals and _normalize(gvals[0]) == _normalize(pvals[0]):
                    exact_hits[field] += 1
                if gvals and pvals and _partial_match(gvals[0], pvals[0]):
                    partial_hits[field] += 1

    token_f1 = float(f1_score(token_true, token_pred, average="macro", zero_division=0))
    field_exact = {
        f: exact_hits[f] / totals[f] for f in totals
    }
    field_partial = {
        f: partial_hits[f] / totals[f] for f in totals
    }
    overall_exact = float(np.mean(list(field_exact.values()))) if field_exact else 0.0
    overall_partial = float(np.mean(list(field_partial.values()))) if field_partial else 0.0

    noisy_metrics = None
    if noisy_prepared and (noisy_prepared / f"{split}.jsonl").exists():
        noisy_metrics = evaluate(model_dir, noisy_prepared, cfg, split=split, noisy_prepared=None)
        noisy_metrics = {
            "token_macro_f1": noisy_metrics["token_macro_f1"],
            "field_exact_mean": noisy_metrics["field_exact_mean"],
            "field_partial_mean": noisy_metrics["field_partial_mean"],
        }

    report = {
        "split": split,
        "n": len(rows),
        "token_macro_f1": token_f1,
        "field_exact_mean": overall_exact,
        "field_partial_mean": overall_partial,
        "field_exact": field_exact,
        "field_partial": field_partial,
        "hard_fields": {f: field_partial.get(f, 0.0) for f in hard_fields},
        "noisy_stress": noisy_metrics,
        "model_dir": str(model_dir),
    }
    write_json(cfg.evaluation_reports_dir / "extraction_report.json", report)

    failure_md = [
        "# Extraction failure modes",
        "",
        "Observed / expected hard fields on synthetic forms:",
        "",
        "- **Dates (`date_of_loss`, `effective_date`)**: OCR digit confusions (0/O, 1/l) and format variation.",
        "- **Dollar amounts (`estimated_damage`, `deductible`, `reserve_set`)**: commas, `$` glyphs, and OCR substitutions.",
        "- **Free-text location / narrative-adjacent values**: long spans bleed into neighboring fields under BIO labeling.",
        "- **Noisy variants**: token F1 and field exact-match drop vs clean renders; partial match remains more stable.",
        "",
        "## Measured hard-field partial match",
        "",
    ]
    for f in hard_fields:
        failure_md.append(f"- `{f}`: {report['hard_fields'].get(f, 0.0):.3f}")
    if noisy_metrics:
        failure_md.extend(
            [
                "",
                "## Noisy stress summary",
                "",
                f"- token_macro_f1: {noisy_metrics['token_macro_f1']:.3f}",
                f"- field_exact_mean: {noisy_metrics['field_exact_mean']:.3f}",
                f"- field_partial_mean: {noisy_metrics['field_partial_mean']:.3f}",
            ]
        )
    (cfg.evaluation_reports_dir / "failure_modes.md").write_text(
        "\n".join(failure_md) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--noisy-prepared", type=Path, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = Config.load()
    report = evaluate(args.model_dir, args.prepared, cfg, args.split, args.noisy_prepared)
    print(
        json.dumps(
            {
                "token_macro_f1": report["token_macro_f1"],
                "field_exact_mean": report["field_exact_mean"],
                "field_partial_mean": report["field_partial_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
