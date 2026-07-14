# smol-doc-analyzer

A small, locally-deployable LLM pipeline for insurance document classification, extraction, and memo generation — built as a cost-efficient alternative to frontier enterprise models for high-volume document intake.

## Why this exists

Insurance operations (claims intake, underwriting submissions, policy servicing) generate huge volumes of structured-but-messy documents — ACORD forms, loss notices, adjuster reports, correspondence. Processing these with frontier API models (GPT-4/Claude-class) works, but at volume it's expensive and requires sending documents to a third party. This project explores whether a small, specialized, **locally-hosted** pipeline can match "good enough" accuracy at a fraction of the cost, fully in-house.

Rather than one large model trying to do everything, this pipeline splits the problem into three specialized components:

1. **Classification** — a small fine-tuned encoder (DeBERTa-v3 class) identifies document type and category
2. **Extraction** — a layout-aware model (LayoutLMv3/Donut class) pulls structured fields from forms, including scanned/OCR'd documents
3. **Summarization** — a small fine-tuned generative LLM (7–8B, quantized) writes adjuster-style memos and analysis from the extracted content

All three run offline, on modest hardware, with no per-token API costs.

## Status

Phases 0–3 and Phase 5 (chained inference orchestrator) implemented:

- Characteristic profiles from public document/layout/legal-style priors
- Synthetic skeleton → document → memo → OCR-noise pipeline (template fallback; optional OpenRouter LLM)
- Document-type classifier train/eval (text DeBERTa + optional ViT image path)
- Field extraction train/eval with noisy stress reporting
- **Single-action analysis chain**: to_markdown → classify → extract → vision_llm → summarize (`src/pipeline/`)
- PNG/PDF → structured markdown before LLM stages (token + context optimization)

Phase 4 (fine-tuned summarizer LoRA) still pending — the chain currently uses a
template memo grounded in upstream extraction/vision outputs, with a hook for a
local generative model when configured.

## Data disclosure

This project uses **no real insurance company data**. All training data is either:

- Publicly available (ACORD form templates, public document-layout datasets, public insurance distribution shapes, legal writing used for **vocabulary/reasoning style only**), or
- Synthetically generated (fictional claims, documents, and memos produced from randomized skeletons)

Legal corpora never become classification labels. See [docs/data_provenance.md](docs/data_provenance.md).

Every synthetic record is logged in `data/provenance_log.jsonl` with its generation source.

## Quickstart

```bash
# install
pip install -e ".[dev]"

# create local secrets file (gitignored) — paste keys into .env
python scripts/setup_env.py
# then edit .env:
#   OPENROUTER_API_KEY=...   # https://openrouter.ai/keys
#   WANDB_API_KEY=...        # https://wandb.ai/authorize
#   HF_TOKEN=...             # optional, https://huggingface.co/settings/tokens
python scripts/setup_env.py --status   # confirms which secrets are set (never prints values)

# refresh profiles (optional Hub ingest: add --ingest)
python -m src.generation.run_seed_pipeline --n 240

# prepare + train classifier (smoke uses DistilBERT on CPU)
python -m src.classification.prepare_dataset --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl
python -m src.classification.train_classifier --prepared data/synthetic/documents/classification_prepared --smoke
python -m src.classification.eval --model-dir models/classifier_smoke --prepared data/synthetic/documents/classification_prepared

# classical baseline: TF-IDF + Random Forest on typed text + handwriting/OCR noise
python -m src.classification.train_random_forest
# or interactively:
#   pip install -e ".[notebooks]"
#   jupyter notebook notebooks/random_forest_text_handwriting_classification.ipynb

# ViT document-image classifier (Kaggle/HF-style image classification on rendered pages)
python -m src.extraction.render_forms --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl --out data/synthetic/documents/rendered
python -m src.classification.prepare_image_dataset --in data/synthetic/documents/rendered/rendered.jsonl
python -m src.classification.train_vit_classifier --prepared data/synthetic/documents/rendered/vit_classification_prepared --smoke
python -m src.classification.eval_vit --model-dir models/vit_classifier_smoke --prepared data/synthetic/documents/rendered/vit_classification_prepared

# render forms + train extractor (smoke path)
python -m src.extraction.render_forms --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl --out data/synthetic/documents/rendered
python -m src.extraction.prepare_dataset --in data/synthetic/documents/rendered/rendered.jsonl
python -m src.extraction.train_extractor --prepared data/synthetic/documents/rendered/extraction_prepared --smoke
python -m src.extraction.eval --model-dir models/extractor_smoke --prepared data/synthetic/documents/rendered/extraction_prepared

# --- one action: full document analysis chain ---
# PNG/PDF are converted to structured markdown before LLM stages (token-efficient).
python -m src.pipeline.orchestrator \
  --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl \
  --out data/pipeline/analysis.jsonl \
  --vision --limit 20

python -m src.pipeline.orchestrator --image path/to/scan.png --vision
python -m src.pipeline.orchestrator --pdf path/to/claim.pdf --vision

python -m src.pipeline.batch_runner \
  --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl \
  --out-dir data/pipeline/batch_demo \
  --vision --limit 20

# ad-hoc single document
python -m src.pipeline.orchestrator --vision --text "LOSS NOTICE\nClaim Number: CLM-1\nDate of Loss: 2024-01-15\n..."

# --- Discord bot (Chloride) ---
# Slash commands: /analyze /analyze_url /status /help /ping
# Agent tool: analyze_insurance_document → local pipeline
pip install -e ".[discord]"
python scripts/setup_env.py   # set DISCORD_TOKEN (+ OPENROUTER_API_KEY)
cp discord/smol-doc-analyzer/config.yaml.example discord/smol-doc-analyzer/config.yaml
python -m src.discord_bot
# See discord/smol-doc-analyzer/README.md

# full-scale generation (when ready)
python -m src.generation.skeleton_sampler --n 5000 --out data/synthetic/skeletons/
python -m src.generation.stage_a_document_gen --in data/synthetic/skeletons/skeletons_n5000_seed42.jsonl
python -m src.generation.stage_b_memo_gen --in data/synthetic/documents/documents_from_skeletons_n5000_seed42.jsonl
```

For GPU training, omit `--smoke` and use the default DeBERTa-v3 / ViT / LayoutLMv3 model names.

On high-RAM local hosts, point `VISION_LLM_MODEL_PATH` at a downloaded Qwen2-VL
(or similar) checkpoint and set `VISION_LLM_LOAD=1` to refine extraction from
page images inside the same chain.

## Repository structure

See [docs/architecture.md](docs/architecture.md).

## Discord bot (Chloride)

Optional Discord front-end via [Chloride](https://github.com/S4IL21/chloride)
(Coral AI agent). Install the `discord` extra, set `DISCORD_TOKEN`, and run
`python -m src.discord_bot`.

Users can:
- run **slash commands** `/analyze`, `/analyze_url`, `/status`, `/help`, `/ping`
- or chat with the bot (mention / `--` prefix) so the agent calls `analyze_insurance_document`

Details: [discord/smol-doc-analyzer/README.md](discord/smol-doc-analyzer/README.md).

## Evaluation

Per-model reports land in `evaluation/reports/` (`classification_report.*`,
`vit_classification_report.*`, `random_forest_classification_report.md`,
`extraction_report.json`, `failure_modes.md`).

### Frontier vs. local harness (Phase 7)

Compare Anthropic / OpenAI (via OpenRouter) against local pipeline models on the
same held-out set across classification, extraction, and memo generation:

```bash
# plan calls without spending API budget
python -m evaluation.eval_harness \
  --eval-set data/eval/eval_set.jsonl \
  --tasks classification extraction memo_generation \
  --backends anthropic openai local \
  --n-samples 50 \
  --output-dir evaluation/results/eval_run_2026-07-13 \
  --dry-run

# live run (requires OPENROUTER_API_KEY for frontier backends)
python -m evaluation.eval_harness \
  --eval-set data/eval/eval_set.jsonl \
  --backends anthropic openai local \
  --output-dir evaluation/results/eval_run_2026-07-13
```

Outputs: `eval_results.jsonl` (source of truth) + `eval_results.csv` (cost-model
spreadsheet feed). Pricing lives in `evaluation/pricing.yaml`.

## Experiment tracking (Weights & Biases)

Training, evaluation, and the seed generation pipeline log to [Weights & Biases](https://wandb.ai) by default:

- **Train**: Hugging Face Trainer metrics (`loss`, eval accuracy / F1), run config, `train_meta.json` artifacts (text DeBERTa, **ViT image**, and LayoutLMv3 extractors)
- **Eval**: summary metrics, per-class / field tables, confusion matrix (classifier + ViT), report + failure-mode artifacts
- **Generation**: stage progress and output path summaries for `run_seed_pipeline`

```bash
# copy env and set your key (https://wandb.ai/authorize)
cp .env.example .env
# WANDB_API_KEY=...  WANDB_PROJECT=smol-doc-analyzer

# offline / no key still works (local wandb/ cache)
WANDB_MODE=offline python -m src.classification.train_classifier --prepared ... --smoke

# disable for a single invocation
python -m src.classification.train_classifier --prepared ... --smoke --no-wandb
```

Useful flags on train/eval/seed CLIs: `--wandb`, `--no-wandb`, `--wandb-project`, `--wandb-run-name`.
Set `WANDB_MODE=disabled` or `WANDB_DISABLED=true` to turn tracking off globally.

## Experiment tracking (Weights & Biases)

Training, evaluation, and the seed generation pipeline log to [Weights & Biases](https://wandb.ai) by default:

- **Train**: Hugging Face Trainer metrics (`loss`, eval accuracy / F1), run config, `train_meta.json` artifacts
- **Eval**: summary metrics, per-class / field tables, confusion matrix (classifier), report + failure-mode artifacts
- **Generation**: stage progress and output path summaries for `run_seed_pipeline`

```bash
# copy env and set your key (https://wandb.ai/authorize)
python scripts/setup_env.py
# WANDB_API_KEY=...  WANDB_PROJECT=smol-doc-analyzer

# offline / no key still works (local wandb/ cache)
WANDB_MODE=offline python -m src.classification.train_classifier --prepared ... --smoke

# disable for a single invocation
python -m src.classification.train_classifier --prepared ... --smoke --no-wandb
```

Useful flags on train/eval/seed CLIs: `--wandb`, `--no-wandb`, `--wandb-project`, `--wandb-run-name`.
Set `WANDB_MODE=disabled` or `WANDB_DISABLED=true` to turn tracking off globally.

## License

TBD — add before making repository public.

## Acknowledgments

Document taxonomy references publicly available ACORD form structures (acord.org). This project is not affiliated with or endorsed by ACORD.

Discord agent integration uses [Chloride](https://github.com/S4IL21/chloride) (fork of [Coral](https://github.com/uukelele/coral)), MIT licensed.
