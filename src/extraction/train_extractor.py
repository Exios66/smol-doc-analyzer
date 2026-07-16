"""Fine-tune LayoutLMv3 (or a smoke text-only token classifier) for field extraction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json
from src.utils.provenance import ProvenanceRecord, log_provenance
from src.utils.wandb_utils import WandbSettings, add_wandb_cli_flags, settings_from_args, start_run

logger = logging.getLogger(__name__)


def train(
    prepared_dir: Path,
    cfg: Config,
    model_name: str = "microsoft/layoutlmv3-base",
    output_dir: Path | None = None,
    max_steps: int | None = None,
    smoke: bool = False,
    wandb_settings: WandbSettings | None = None,
    wandb_run_name: str | None = None,
) -> Path:
    import numpy as np
    from datasets import Dataset
    from PIL import Image
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    label2id = {k: int(v) for k, v in read_json(prepared_dir / "label2id.json").items()}
    id2label = {v: k for k, v in label2id.items()}
    train_rows = load_jsonl(prepared_dir / "train.jsonl")
    val_rows = load_jsonl(prepared_dir / "val.jsonl") if (prepared_dir / "val.jsonl").exists() else []

    if smoke:
        train_rows = train_rows[:24]
        val_rows = val_rows[:8] or train_rows[:4]
        # LayoutLMv3 is heavy for CPU smoke; use a tiny token classifier on text only
        model_name = "distilbert-base-uncased"
        max_steps = max_steps or 20

    if not train_rows:
        raise RuntimeError("No extraction train rows")

    out = output_dir or (cfg.models_dir / ("extractor_smoke" if smoke else "extractor"))
    out.mkdir(parents=True, exist_ok=True)

    run_config = {
        "task": "extraction",
        "model_name": model_name,
        "prepared_dir": str(prepared_dir),
        "smoke": smoke,
        "max_steps": max_steps,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_labels": len(label2id),
        "output_dir": str(out),
    }
    run_name = wandb_run_name or f"ext-{'smoke' if smoke else 'train'}-{out.name}"

    with start_run(
        name=run_name,
        job_type="train",
        config=run_config,
        tags=["extraction", "smoke" if smoke else "full"],
        settings=wandb_settings,
    ) as wb:
        metrics: dict = {}
        if smoke or "distilbert" in model_name:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                DataCollatorForTokenClassification,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

            def tokenize_and_align(example):
                tok = tokenizer(
                    example["tokens"],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=256,
                )
                word_ids = tok.word_ids()
                labels = []
                previous_wid = None
                for wid in word_ids:
                    if wid is None:
                        labels.append(-100)
                    elif wid != previous_wid:
                        # First subword of each word keeps the label.
                        labels.append(example["labels"][wid])
                    else:
                        # Continuation subwords are ignored (matches eval alignment).
                        labels.append(-100)
                    previous_wid = wid
                tok["labels"] = labels
                return tok

            train_ds = Dataset.from_list(train_rows).map(tokenize_and_align)
            val_ds = Dataset.from_list(val_rows).map(tokenize_and_align) if val_rows else None
            collator = DataCollatorForTokenClassification(tokenizer)

            def compute_metrics(p):
                logits, labels = p
                preds = np.argmax(logits, axis=-1)
                true_flat, pred_flat = [], []
                for pred_seq, lab_seq in zip(preds, labels):
                    for pr, lb in zip(pred_seq, lab_seq):
                        if lb == -100:
                            continue
                        true_flat.append(lb)
                        pred_flat.append(pr)
                return {
                    "token_macro_f1": float(
                        f1_score(true_flat, pred_flat, average="macro", zero_division=0)
                    )
                }

            args = TrainingArguments(
                output_dir=str(out / "checkpoints"),
                learning_rate=5e-5,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                max_steps=max_steps if max_steps is not None else 200,
                eval_strategy="steps" if val_ds is not None else "no",
                eval_steps=10,
                save_strategy="no",
                logging_steps=5,
                report_to=wb.report_to,
                run_name=run_name,
                seed=42,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=collator,
                processing_class=tokenizer,
                compute_metrics=compute_metrics if val_ds is not None else None,
            )
            train_result = trainer.train()
            trainer.save_model(str(out))
            tokenizer.save_pretrained(str(out))
            metrics = dict(train_result.metrics)
            if val_ds is not None:
                metrics.update(trainer.evaluate())
        else:
            processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
            )

            def preprocess(example):
                image = Image.open(example["image_path"]).convert("RGB")
                encoding = processor(
                    image,
                    example["tokens"],
                    boxes=example["bboxes"],
                    word_labels=example["labels"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                )
                return {k: v.squeeze(0) for k, v in encoding.items()}

            train_ds = Dataset.from_list(train_rows).map(preprocess)
            val_ds = Dataset.from_list(val_rows).map(preprocess) if val_rows else None
            args = TrainingArguments(
                output_dir=str(out / "checkpoints"),
                learning_rate=2e-5,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                max_steps=max_steps if max_steps is not None else 500,
                eval_strategy="no",
                save_strategy="no",
                logging_steps=20,
                report_to=wb.report_to,
                run_name=run_name,
                seed=42,
                remove_unused_columns=False,
            )
            trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
            train_result = trainer.train()
            trainer.save_model(str(out))
            processor.save_pretrained(str(out))
            metrics = dict(train_result.metrics)

        wb.summary({f"train/{k}": v for k, v in metrics.items()})
        wb.log({f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

        meta = {
            "model_name": model_name,
            "smoke": smoke,
            "n_train": len(train_rows),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
        }
        write_json(out / "label2id.json", label2id)
        write_json(out / "train_meta.json", meta)
        wb.log_artifact_files(
            name=f"extractor-train-meta-{out.name}",
            paths=[out / "train_meta.json", out / "label2id.json"],
            artifact_type="model-meta",
            metadata={"smoke": smoke, "model_name": model_name},
        )

        log_provenance(
            cfg.provenance_log_path,
            ProvenanceRecord(
                record_id=f"ext-train-{out.name}",
                stage="extraction_train",
                source=str(prepared_dir),
                prompt_version="extraction_v1",
                model=model_name,
                extra={"out": str(out), "smoke": smoke, "wandb": wb.active},
            ),
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    add_wandb_cli_flags(parser)
    args = parser.parse_args()
    print(
        train(
            args.prepared,
            Config.load(),
            args.model,
            args.out,
            args.max_steps,
            args.smoke,
            wandb_settings=settings_from_args(args),
            wandb_run_name=args.wandb_run_name,
        )
    )


if __name__ == "__main__":
    main()
