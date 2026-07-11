"""Fine-tune a DeBERTa-v3 classifier on synthetic insurance documents."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json
from src.utils.provenance import ProvenanceRecord, log_provenance
from src.utils.wandb_utils import WandbSettings, add_wandb_cli_flags, settings_from_args, start_run

logger = logging.getLogger(__name__)


def _load_split_texts(prepared_dir: Path, split: str) -> list[dict]:
    path = prepared_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def train(
    prepared_dir: Path,
    cfg: Config,
    model_name: str = "microsoft/deberta-v3-base",
    output_dir: Path | None = None,
    max_steps: int | None = None,
    epochs: float = 2.0,
    smoke: bool = False,
    wandb_settings: WandbSettings | None = None,
    wandb_run_name: str | None = None,
) -> Path:
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    label2id = read_json(prepared_dir / "label2id.json")
    id2label = {int(v): k for k, v in label2id.items()}
    label2id = {k: int(v) for k, v in label2id.items()}

    train_rows = _load_split_texts(prepared_dir, "train")
    val_rows = _load_split_texts(prepared_dir, "val")
    if smoke:
        train_rows = train_rows[:64]
        val_rows = val_rows[:32] or train_rows[:16]
        model_name = "distilbert-base-uncased"
        epochs = 1.0
        max_steps = max_steps or 30

    if not train_rows:
        raise RuntimeError(f"No train rows in {prepared_dir}")

    out = output_dir or (cfg.models_dir / ("classifier_smoke" if smoke else "classifier"))
    out.mkdir(parents=True, exist_ok=True)

    run_config = {
        "task": "classification",
        "model_name": model_name,
        "prepared_dir": str(prepared_dir),
        "smoke": smoke,
        "epochs": epochs,
        "max_steps": max_steps,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_labels": len(label2id),
        "learning_rate": 2e-5,
        "output_dir": str(out),
    }
    run_name = wandb_run_name or f"clf-{'smoke' if smoke else 'train'}-{out.name}"

    with start_run(
        name=run_name,
        job_type="train",
        config=run_config,
        tags=["classification", "smoke" if smoke else "full"],
        settings=wandb_settings,
    ) as wb:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        def tok(batch):
            return tokenizer(batch["text"], truncation=True, max_length=512)

        train_ds = Dataset.from_list(train_rows).map(tok, batched=True)
        val_ds = Dataset.from_list(val_rows).map(tok, batched=True) if val_rows else None
        train_ds = train_ds.rename_column("label_id", "labels")
        if val_ds is not None:
            val_ds = val_ds.rename_column("label_id", "labels")

        keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
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
                "macro_f1": float(f1_score(labels, preds, average="macro")),
            }

        args = TrainingArguments(
            output_dir=str(out / "checkpoints"),
            learning_rate=2e-5,
            per_device_train_batch_size=8 if not smoke else 4,
            per_device_eval_batch_size=8 if not smoke else 4,
            num_train_epochs=epochs,
            max_steps=max_steps if max_steps is not None else -1,
            eval_strategy="steps" if val_ds is not None else "no",
            eval_steps=20 if smoke else 100,
            save_strategy="no",
            logging_steps=10,
            report_to=wb.report_to,
            run_name=run_name,
            seed=42,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics if val_ds is not None else None,
        )
        train_result = trainer.train()
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))

        metrics = dict(train_result.metrics)
        if val_ds is not None:
            metrics.update(trainer.evaluate())
        wb.summary({f"train/{k}": v for k, v in metrics.items()})
        wb.log({f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

        meta = {
            "model_name": model_name,
            "smoke": smoke,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "max_steps": max_steps,
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
        }
        write_json(out / "label2id.json", label2id)
        write_json(out / "train_meta.json", meta)
        wb.log_artifact_files(
            name=f"classifier-train-meta-{out.name}",
            paths=[out / "train_meta.json", out / "label2id.json"],
            artifact_type="model-meta",
            metadata={"smoke": smoke, "model_name": model_name},
        )

        log_provenance(
            cfg.provenance_log_path,
            ProvenanceRecord(
                record_id=f"clf-train-{out.name}",
                stage="classification_train",
                source=str(prepared_dir),
                prompt_version="classification_v1",
                model=model_name,
                extra={"out": str(out), "smoke": smoke, "wandb": wb.active},
            ),
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=2.0)
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
