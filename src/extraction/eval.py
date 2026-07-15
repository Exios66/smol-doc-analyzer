"""Field-level extraction evaluation + failure mode notes."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json
from src.utils.wandb_utils import WandbSettings, add_wandb_cli_flags, settings_from_args, start_run


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


def _score_field_entities(
    gvals: list[str], pvals: list[str]
) -> tuple[int, int, int]:
    """
    Greedy 1:1 entity matching.

    Returns (exact_hits, partial_hits, totals) where totals is max(len(g), len(p))
    so unmatched preds count as misses too.
    """
    totals = max(len(gvals), len(pvals), 1 if (gvals or pvals) else 0)
    if totals == 0:
        return 0, 0, 0
    used: set[int] = set()
    exact = 0
    partial = 0
    for g in gvals:
        best_j = None
        best_exact = False
        for j, p in enumerate(pvals):
            if j in used:
                continue
            if _normalize(g) == _normalize(p):
                best_j = j
                best_exact = True
                break
            if best_j is None and _partial_match(g, p):
                best_j = j
                best_exact = False
        if best_j is not None:
            used.add(best_j)
            if best_exact:
                exact += 1
                partial += 1
            else:
                partial += 1
    return exact, partial, max(len(gvals), len(pvals), 1)


def _is_layoutlm_checkpoint(model_dir: Path) -> bool:
    meta_path = model_dir / "train_meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        name = str(meta.get("model_name") or "")
        if "layoutlm" in name.lower() and not meta.get("smoke"):
            return True
    # Processor files indicate a vision+layout checkpoint.
    return (model_dir / "preprocessor_config.json").exists() or (
        model_dir / "processor_config.json"
    ).exists()


def _evaluate_text_checkpoint(
    model_dir: Path,
    rows: list[dict],
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> tuple[list[int], list[int], dict[str, int], dict[str, int], dict[str, int]]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    token_true, token_pred = [], []
    exact_hits: dict[str, int] = defaultdict(int)
    partial_hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

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
            seen: set[int] = set()
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
                ex, part, tot = _score_field_entities(gvals, pvals)
                exact_hits[field] += ex
                partial_hits[field] += part
                totals[field] += tot

    return token_true, token_pred, exact_hits, partial_hits, totals


def _evaluate_layout_checkpoint(
    model_dir: Path,
    rows: list[dict],
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> tuple[list[int], list[int], dict[str, int], dict[str, int], dict[str, int]]:
    import torch
    from PIL import Image
    from transformers import AutoModelForTokenClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_dir), apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    token_true, token_pred = [], []
    exact_hits: dict[str, int] = defaultdict(int)
    partial_hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            encoding = processor(
                image,
                row["tokens"],
                boxes=row["bboxes"],
                word_labels=row["labels"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            labels = encoding.pop("labels")[0].tolist()
            outputs = model(**encoding)
            pred_ids = torch.argmax(outputs.logits[0], dim=-1).tolist()

            aligned_pred: list[str] = []
            gold: list[str] = []
            pred_tokens: list[str] = []
            # LayoutLMv3 word alignment: use first subword per word via word_ids when present.
            word_ids = encoding.word_ids(batch_index=0) if hasattr(encoding, "word_ids") else None
            if word_ids is None:
                for lab, pred in zip(labels, pred_ids):
                    if lab == -100:
                        continue
                    gold.append(id2label.get(int(lab), "O"))
                    aligned_pred.append(id2label.get(int(pred), "O"))
                pred_tokens = list(row["tokens"][: len(gold)])
            else:
                seen: set[int] = set()
                aligned_pred = ["O"] * len(row["tokens"])
                for idx, wid in enumerate(word_ids):
                    if wid is None or wid in seen:
                        continue
                    seen.add(wid)
                    aligned_pred[wid] = id2label.get(int(pred_ids[idx]), "O")
                gold = [id2label.get(int(i), "O") for i in row["labels"]]
                pred_tokens = row["tokens"]

            for g, p in zip(gold, aligned_pred):
                token_true.append(label2id.get(g, 0))
                token_pred.append(label2id.get(p, 0))

            gold_ents = _decode_entities(pred_tokens, gold)
            pred_ents = _decode_entities(pred_tokens, aligned_pred)
            fields = set(gold_ents) | set(pred_ents)
            for field in fields:
                gvals = gold_ents.get(field, [])
                pvals = pred_ents.get(field, [])
                ex, part, tot = _score_field_entities(gvals, pvals)
                exact_hits[field] += ex
                partial_hits[field] += part
                totals[field] += tot

    return token_true, token_pred, exact_hits, partial_hits, totals


def _evaluate_core(
    model_dir: Path,
    prepared_dir: Path,
    split: str = "test",
) -> dict:
    rows = load_jsonl(prepared_dir / f"{split}.jsonl")
    label2id = {k: int(v) for k, v in read_json(model_dir / "label2id.json").items()}
    id2label = {v: k for k, v in label2id.items()}

    if _is_layoutlm_checkpoint(model_dir):
        token_true, token_pred, exact_hits, partial_hits, totals = _evaluate_layout_checkpoint(
            model_dir, rows, id2label, label2id
        )
    else:
        token_true, token_pred, exact_hits, partial_hits, totals = _evaluate_text_checkpoint(
            model_dir, rows, id2label, label2id
        )

    hard_fields = ["date_of_loss", "estimated_damage", "deductible", "reserve_set", "location"]
    token_f1 = float(f1_score(token_true, token_pred, average="macro", zero_division=0))
    field_exact = {f: exact_hits[f] / totals[f] for f in totals}
    field_partial = {f: partial_hits[f] / totals[f] for f in totals}
    overall_exact = float(np.mean(list(field_exact.values()))) if field_exact else 0.0
    overall_partial = float(np.mean(list(field_partial.values()))) if field_partial else 0.0

    return {
        "split": split,
        "n": len(rows),
        "token_macro_f1": token_f1,
        "field_exact_mean": overall_exact,
        "field_partial_mean": overall_partial,
        "field_exact": field_exact,
        "field_partial": field_partial,
        "hard_fields": {f: field_partial.get(f, 0.0) for f in hard_fields},
        "model_dir": str(model_dir),
        "_hard_fields_list": hard_fields,
    }


def evaluate(
    model_dir: Path,
    prepared_dir: Path,
    cfg: Config,
    split: str = "test",
    noisy_prepared: Path | None = None,
    wandb_settings: WandbSettings | None = None,
    wandb_run_name: str | None = None,
) -> dict:
    report = _evaluate_core(model_dir, prepared_dir, split=split)
    hard_fields = report.pop("_hard_fields_list")

    noisy_metrics = None
    if noisy_prepared and (noisy_prepared / f"{split}.jsonl").exists():
        noisy = _evaluate_core(model_dir, noisy_prepared, split=split)
        noisy_metrics = {
            "token_macro_f1": noisy["token_macro_f1"],
            "field_exact_mean": noisy["field_exact_mean"],
            "field_partial_mean": noisy["field_partial_mean"],
        }

    report["noisy_stress"] = noisy_metrics
    cfg.evaluation_reports_dir.mkdir(parents=True, exist_ok=True)
    report_json = cfg.evaluation_reports_dir / "extraction_report.json"
    write_json(report_json, report)

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
    failure_path = cfg.evaluation_reports_dir / "failure_modes.md"
    failure_path.write_text("\n".join(failure_md) + "\n", encoding="utf-8")

    run_name = wandb_run_name or f"ext-eval-{Path(model_dir).name}-{split}"
    with start_run(
        name=run_name,
        job_type="eval",
        config={
            "task": "extraction_eval",
            "model_dir": str(model_dir),
            "prepared_dir": str(prepared_dir),
            "noisy_prepared": str(noisy_prepared) if noisy_prepared else None,
            "split": split,
            "n": report["n"],
        },
        tags=["extraction", "eval", split],
        settings=wandb_settings,
    ) as wb:
        wb.summary(
            {
                "token_macro_f1": report["token_macro_f1"],
                "field_exact_mean": report["field_exact_mean"],
                "field_partial_mean": report["field_partial_mean"],
                "n": report["n"],
                "split": split,
            }
        )
        wb.log(
            {
                "eval/token_macro_f1": report["token_macro_f1"],
                "eval/field_exact_mean": report["field_exact_mean"],
                "eval/field_partial_mean": report["field_partial_mean"],
                "eval/n": report["n"],
            }
        )
        field_rows = [
            [
                field,
                report["field_exact"].get(field, 0.0),
                report["field_partial"].get(field, 0.0),
            ]
            for field in sorted(set(report["field_exact"]) | set(report["field_partial"]))
        ]
        wb.log_table(
            "eval/field_scores",
            ["field", "exact", "partial"],
            field_rows,
        )
        wb.log_table(
            "eval/hard_fields",
            ["field", "partial"],
            [[f, report["hard_fields"].get(f, 0.0)] for f in hard_fields],
        )
        if noisy_metrics:
            wb.log(
                {
                    "eval/noisy_token_macro_f1": noisy_metrics["token_macro_f1"],
                    "eval/noisy_field_exact_mean": noisy_metrics["field_exact_mean"],
                    "eval/noisy_field_partial_mean": noisy_metrics["field_partial_mean"],
                }
            )
            wb.summary({f"noisy/{k}": v for k, v in noisy_metrics.items()})
        wb.log_artifact_files(
            name=f"extraction-report-{Path(model_dir).name}-{split}",
            paths=[report_json, failure_path],
            artifact_type="evaluation",
            metadata={
                "token_macro_f1": report["token_macro_f1"],
                "field_exact_mean": report["field_exact_mean"],
                "split": split,
            },
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--noisy-prepared", type=Path, default=None)
    parser.add_argument("--split", default="test")
    add_wandb_cli_flags(parser)
    args = parser.parse_args()
    cfg = Config.load()
    report = evaluate(
        args.model_dir,
        args.prepared,
        cfg,
        args.split,
        args.noisy_prepared,
        wandb_settings=settings_from_args(args),
        wandb_run_name=args.wandb_run_name,
    )
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
