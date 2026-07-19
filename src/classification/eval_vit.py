"""Evaluate the ViT document-image classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json
from src.utils.wandb_utils import WandbSettings, add_wandb_cli_flags, settings_from_args, start_run


def evaluate(
    model_dir: Path,
    prepared_dir: Path,
    cfg: Config,
    split: str = "test",
    wandb_settings: WandbSettings | None = None,
    wandb_run_name: str | None = None,
) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    rows = load_jsonl(prepared_dir / f"{split}.jsonl")
    if not rows:
        raise RuntimeError(f"No ViT rows for split={split}")

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageClassification.from_pretrained(str(model_dir))
    model.eval()
    label2id = read_json(model_dir / "label2id.json")
    id2label = {int(v): k for k, v in label2id.items()}
    labels_order = [id2label[i] for i in range(len(id2label))]

    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba: list[list[float]] = []
    with torch.no_grad():
        for row in rows:
            image = Image.open(row["image_path"]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()
            pred = int(torch.argmax(logits, dim=-1).item())
            y_pred.append(pred)
            y_true.append(int(row["label_id"]))
            y_proba.append([float(p) for p in probs])

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_order)))).tolist()
    per_class = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels_order))),
        target_names=labels_order,
        output_dict=True,
        zero_division=0,
    )

    auc_ovr = auc_ovo = None
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(y_true)) >= 2:
            proba = np.asarray(y_proba, dtype=float)
            auc_ovr = float(
                roc_auc_score(y_true, proba, multi_class="ovr", labels=list(range(len(labels_order))))
            )
            auc_ovo = float(
                roc_auc_score(y_true, proba, multi_class="ovo", labels=list(range(len(labels_order))))
            )
    except ValueError:
        auc_ovr = auc_ovo = None
    if auc_ovr is not None and (np.isnan(auc_ovr) or np.isinf(auc_ovr)):
        auc_ovr = None
    if auc_ovo is not None and (np.isnan(auc_ovo) or np.isinf(auc_ovo)):
        auc_ovo = None

    pairs = []
    for i, row_cm in enumerate(cm):
        for j, count in enumerate(row_cm):
            if i != j and count > 0:
                pairs.append(
                    {"true": labels_order[i], "pred": labels_order[j], "count": int(count)}
                )
    pairs.sort(key=lambda x: -x["count"])

    tax = yaml.safe_load(cfg.taxonomy_path.read_text(encoding="utf-8"))
    report = {
        "split": split,
        "n": len(rows),
        "modality": "image",
        "architecture": "vit",
        "accuracy": acc,
        "auc_ovr": auc_ovr,
        "auc_ovo": auc_ovo,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "labels": labels_order,
        "confusion_matrix": cm,
        "per_class": per_class,
        "top_confusion_pairs": pairs[:15],
        "taxonomy_version_note": "acord_form_categories.yaml v1",
        "taxonomy_category_count": len(tax["categories"]),
        "model_dir": str(model_dir),
    }

    cfg.evaluation_reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = cfg.evaluation_reports_dir / "vit_classification_report.json"
    write_json(out_json, report)
    md = [
        "# ViT document image classification report",
        "",
        f"- Split: `{split}`",
        f"- N: {len(rows)}",
        f"- Accuracy: **{acc:.4f}**",
        f"- AUC (OVR): **{auc_ovr:.4f}**" if auc_ovr is not None else "- AUC (OVR): —",
        f"- AUC (OVO): **{auc_ovo:.4f}**" if auc_ovo is not None else "- AUC (OVO): —",
        f"- Macro F1: **{macro_f1:.4f}**",
        f"- Micro F1: **{micro_f1:.4f}**",
        f"- Weighted F1: **{weighted_f1:.4f}**",
        "",
        "## Top confusion pairs",
        "",
    ]
    if pairs:
        for p in pairs[:10]:
            md.append(f"- {p['true']} → {p['pred']}: {p['count']}")
    else:
        md.append("- None")
    md_path = cfg.evaluation_reports_dir / "vit_classification_report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    run_name = wandb_run_name or f"vit-eval-{Path(model_dir).name}-{split}"
    with start_run(
        name=run_name,
        job_type="eval",
        config={
            "task": "document_image_classification_eval",
            "modality": "image",
            "architecture": "vit",
            "model_dir": str(model_dir),
            "prepared_dir": str(prepared_dir),
            "split": split,
            "n": len(rows),
        },
        tags=["classification", "vit", "image", "eval", split],
        settings=wandb_settings,
    ) as wb:
        wb.summary({"accuracy": acc, "macro_f1": macro_f1, "n": len(rows), "split": split})
        wb.log({"eval/accuracy": acc, "eval/macro_f1": macro_f1, "eval/n": len(rows)})
        wb.log_confusion_matrix(
            key="eval/confusion_matrix",
            y_true=y_true,
            y_pred=y_pred,
            class_names=labels_order,
        )
        per_class_rows = []
        for label in labels_order:
            stats = per_class.get(label, {})
            if isinstance(stats, dict):
                per_class_rows.append(
                    [
                        label,
                        stats.get("precision", 0.0),
                        stats.get("recall", 0.0),
                        stats.get("f1-score", 0.0),
                        stats.get("support", 0),
                    ]
                )
        wb.log_table(
            "eval/per_class",
            ["label", "precision", "recall", "f1", "support"],
            per_class_rows,
        )
        if pairs:
            wb.log_table(
                "eval/top_confusion_pairs",
                ["true", "pred", "count"],
                [[p["true"], p["pred"], p["count"]] for p in pairs[:15]],
            )
        wb.log_artifact_files(
            name=f"vit-classification-report-{Path(model_dir).name}-{split}",
            paths=[out_json, md_path],
            artifact_type="evaluation",
            metadata={"accuracy": acc, "macro_f1": macro_f1, "split": split, "modality": "image"},
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    add_wandb_cli_flags(parser)
    args = parser.parse_args()
    cfg = Config.load()
    report = evaluate(
        args.model_dir,
        args.prepared,
        cfg,
        args.split,
        wandb_settings=settings_from_args(args),
        wandb_run_name=args.wandb_run_name,
    )
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
