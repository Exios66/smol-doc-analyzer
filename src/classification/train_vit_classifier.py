"""Fine-tune a Vision Transformer for document image classification.

Pattern follows the common Hugging Face / Kaggle ViT document-image workflow:
``AutoImageProcessor`` + ``AutoModelForImageClassification`` + ``Trainer``,
with Weights & Biases run logging when enabled.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json
from src.utils.provenance import ProvenanceRecord, log_provenance
from src.utils.wandb_utils import WandbSettings, add_wandb_cli_flags, settings_from_args, start_run

logger = logging.getLogger(__name__)

DEFAULT_VIT = "google/vit-base-patch16-224-in21k"
SMOKE_VIT = "facebook/deit-tiny-patch16-224"


def _load_split(prepared_dir: Path, split: str) -> list[dict]:
    path = prepared_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def train(
    prepared_dir: Path,
    cfg: Config,
    model_name: str = DEFAULT_VIT,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    epochs: float = 3.0,
    smoke: bool = False,
    wandb_settings: WandbSettings | None = None,
    wandb_run_name: str | None = None,
) -> Path:
    import numpy as np
    import torch
    from datasets import Dataset
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainingArguments,
    )

    label2id = {k: int(v) for k, v in read_json(prepared_dir / "label2id.json").items()}
    id2label = {v: k for k, v in label2id.items()}

    train_rows = _load_split(prepared_dir, "train")
    val_rows = _load_split(prepared_dir, "val")
    if smoke:
        train_rows = train_rows[:32]
        val_rows = val_rows[:16] or train_rows[:8]
        model_name = SMOKE_VIT
        epochs = 1.0
        max_steps = max_steps or 20

    if not train_rows:
        raise RuntimeError(f"No ViT train rows in {prepared_dir}")

    out = output_dir or (cfg.models_dir / ("vit_classifier_smoke" if smoke else "vit_classifier"))
    out.mkdir(parents=True, exist_ok=True)

    run_config = {
        "task": "document_image_classification",
        "modality": "image",
        "model_name": model_name,
        "prepared_dir": str(prepared_dir),
        "smoke": smoke,
        "epochs": epochs,
        "max_steps": max_steps,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_labels": len(label2id),
        "learning_rate": 2e-4 if not smoke else 5e-4,
        "output_dir": str(out),
    }
    run_name = wandb_run_name or f"vit-{'smoke' if smoke else 'train'}-{out.name}"

    with start_run(
        name=run_name,
        job_type="train",
        config=run_config,
        tags=["classification", "vit", "image", "smoke" if smoke else "full"],
        settings=wandb_settings,
    ) as wb:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        def preprocess(example):
            image = Image.open(example["image_path"]).convert("RGB")
            encoding = processor(images=image, return_tensors="pt")
            return {
                "pixel_values": encoding["pixel_values"].squeeze(0),
                "labels": int(example["label_id"]),
            }

        train_ds = Dataset.from_list(train_rows).map(preprocess)
        val_ds = Dataset.from_list(val_rows).map(preprocess) if val_rows else None
        keep = {"pixel_values", "labels"}
        drop_train = [c for c in train_ds.column_names if c not in keep]
        if drop_train:
            train_ds = train_ds.remove_columns(drop_train)
        if val_ds is not None:
            drop_val = [c for c in val_ds.column_names if c not in keep]
            if drop_val:
                val_ds = val_ds.remove_columns(drop_val)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            }

        def collate_fn(batch):
            pixel_values = []
            labels = []
            for item in batch:
                pv = item["pixel_values"]
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                pixel_values.append(pv)
                labels.append(int(item["labels"]))
            return {
                "pixel_values": torch.stack(pixel_values),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        total_steps_hint = max_steps if max_steps is not None and max_steps > 0 else max(
            1, int(len(train_ds) / max(1, (8 if not smoke else 2)) * epochs)
        )
        warmup_steps = max(1, int(0.1 * total_steps_hint)) if not smoke else 0

        args = TrainingArguments(
            output_dir=str(out / "checkpoints"),
            learning_rate=2e-4 if not smoke else 5e-4,
            per_device_train_batch_size=8 if not smoke else 2,
            per_device_eval_batch_size=8 if not smoke else 2,
            num_train_epochs=epochs,
            max_steps=max_steps if max_steps is not None else -1,
            eval_strategy="steps" if val_ds is not None else "no",
            eval_steps=20 if smoke else 50,
            save_strategy="no",
            logging_steps=5 if smoke else 10,
            remove_unused_columns=False,
            report_to=wb.report_to,
            run_name=run_name,
            seed=42,
            warmup_steps=warmup_steps,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            processing_class=processor,
            compute_metrics=compute_metrics if val_ds is not None else None,
        )

        train_result = trainer.train()
        trainer.save_model(str(out))
        processor.save_pretrained(str(out))

        metrics = dict(train_result.metrics)
        if val_ds is not None:
            metrics.update(trainer.evaluate())
        wb.summary({f"train/{k}": v for k, v in metrics.items()})
        wb.log({f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

        meta = {
            "model_name": model_name,
            "modality": "image",
            "architecture": "vit",
            "smoke": smoke,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "max_steps": max_steps,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
        }
        write_json(out / "label2id.json", label2id)
        write_json(out / "train_meta.json", meta)
        wb.log_artifact_files(
            name=f"vit-classifier-train-meta-{out.name}",
            paths=[out / "train_meta.json", out / "label2id.json"],
            artifact_type="model-meta",
            metadata={"smoke": smoke, "model_name": model_name, "modality": "image"},
        )

        log_provenance(
            cfg.provenance_log_path,
            ProvenanceRecord(
                record_id=f"vit-train-{out.name}",
                stage="classification_vit_train",
                source=str(prepared_dir),
                prompt_version="classification_vit_v1",
                model=model_name,
                extra={"out": str(out), "smoke": smoke, "wandb": wb.active},
            ),
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT for insurance document image classification"
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_VIT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--smoke", action="store_true")
    add_wandb_cli_flags(parser)
    args = parser.parse_args()
    cfg = Config.load()
    print(
        train(
            args.prepared,
            cfg,
            args.model,
            args.out,
            args.max_steps,
            args.epochs,
            args.smoke,
            wandb_settings=settings_from_args(args),
            wandb_run_name=args.wandb_run_name,
        )
    )


if __name__ == "__main__":
    main()
