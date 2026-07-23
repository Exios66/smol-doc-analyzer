# smol-doc-analyzer

**Version:** [`1.0.0-beta`](CHANGELOG.md) (`1.0.0b0`)

A small, locally-deployable LLM pipeline for insurance document classification, extraction, and memo generation — built as a cost-efficient alternative to frontier enterprise models for high-volume document intake.

## Why this exists

Insurance operations (claims intake, underwriting submissions, policy servicing) generate huge volumes of structured-but-messy documents — ACORD forms, loss notices, adjuster reports, correspondence. Processing these with frontier API models (GPT-4/Claude-class) works, but at volume it's expensive and requires sending documents to a third party. This project explores whether a small, specialized, **locally-hosted** pipeline can match "good enough" accuracy at a fraction of the cost, fully in-house.

Rather than one large model trying to do everything, this pipeline splits the problem into three specialized components:

1. **Classification** — a small fine-tuned encoder (DeBERTa-v3 class) identifies document type and category
2. **Extraction** — a layout-aware model (LayoutLMv3/Donut class) pulls structured fields from forms, including scanned/OCR'd documents
3. **Summarization** — a small fine-tuned generative LLM (7–8B, quantized) writes adjuster-style memos and analysis from the extracted content

All three run offline, on modest hardware, with no per-token API costs.

## Status

**Package version:** `1.0.0-beta` (`1.0.0b0`) — see [CHANGELOG.md](CHANGELOG.md)
for the incremental history from `0.1.0` through this beta.

Covers Phases 0–3 and 5, DICIE (5b), sample corpus store, Discord bot,
eval/cost harness, and classical/ViT classifiers (Phase 4 summarizer LoRA
still pending):

- Characteristic profiles from public document/layout/legal-style priors
- Synthetic skeleton → document → memo → OCR-noise pipeline (optional OpenRouter LLM; auto-routes to free models on credit exhaustion, else template fallback)
- Document-type classifier train/eval (text DeBERTa + optional ViT image path + TF-IDF Random Forest baseline)
- Field extraction train/eval with noisy stress reporting
- **DICIE (Fig. 1)**: document processing → classification → information extraction → aggregated response (`src/docie/`) for medical bills and salvage claims
- **Single-action analysis chain**: to_markdown → classify → extract → vision_llm → summarize (`src/pipeline/`)
- PNG/PDF → structured markdown before LLM stages (token + context optimization)
- **Sample corpus store**: queryable SQLite house for synthetic medical / salvage docs (`src/storage/`)
- **Discord bot** (Chloride): `/analyze` memo chain plus notes, STT, DJ/vibes, chat
- **Frontier vs. local eval harness** + cost-model spreadsheet

Phase 4 (fine-tuned summarizer LoRA) still pending — the chain currently uses a
template memo grounded in upstream extraction/vision outputs, with a hook for a
local generative model when configured.

**Docs:** [📚 Live Quarto site (Posit Connect Cloud)](https://connect.posit.cloud/jjb-morningstar/content/019f8d0a-2732-fecb-b056-1f69f4451c00) ·
[Usage](docs/usage.md) · [Architecture](docs/architecture.md) ·
[CHANGELOG](CHANGELOG.md) ·
[Local preview / re-publish](docs/how-to/launch-quarto-site.qmd)

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

# --- paper Fig. 1 DICIE pipeline (process → classify → extract → respond) ---
# Docs: src/docie/README.md  ·  docs/docie_pipeline.md
# Medical bills / salvage claims applications from Raj et al.
python -m src.docie \
  --application salvage_claims \
  --text "LETTER OF GUARANTEE
Claim Number: CLM-2024-100200
VIN: 1HGCM82633A004352
Year: 2018
Make: Honda
Model: Accord" \
  --response-only

python -m src.docie --application medical_bills --pdf path/to/bill.pdf
python -m src.docie --application salvage_claims --image path/to/log.png
python -m src.docie \
  --application salvage_claims \
  --in tests/fixtures/sample_docie_documents.jsonl \
  --out data/pipeline/docie/salvage_demo.jsonl

# Optional REST server (paper §VI FastAPI shape)
# pip install -e ".[serve]"
# python -m src.docie.serve --application salvage_claims --port 8080

# --- one action: full document analysis chain (memo path) ---
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
# Docs + notes + STT + DJ/vibes + chat agent
pip install -e ".[discord]"
# optional voice DJ: pip install -e ".[discord,discord-voice]" && brew install ffmpeg
python scripts/setup_env.py   # set DISCORD_TOKEN (+ OPENROUTER_API_KEY; OPENAI_API_KEY for Whisper)
cp discord/smol-doc-analyzer/config.yaml.example discord/smol-doc-analyzer/config.yaml
python -m src.discord_bot
# autostart at login (macOS):
./scripts/install_discord_bot_loginitem.sh
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

## Documentation website (Quarto)

Guides, architecture notes, and notebooks are published as a Quarto website
under [`docs/`](docs/) (`docs/_quarto.yml`).

```bash
# install Quarto CLI: https://quarto.org/docs/get-started/
cd docs && quarto preview           # live local portal (run inside docs/)
./scripts/preview_docs_site.sh      # same from repo root, with a PATH check
cd docs && quarto render            # static build → docs/_site/
```

How-to: [docs/how-to/launch-quarto-site.qmd](docs/how-to/launch-quarto-site.qmd).
Quick start page: [docs/quick-start.qmd](docs/quick-start.qmd).
Notebook portal: [docs/notebooks/index.qmd](docs/notebooks/index.qmd).

## Repository structure

See [docs/architecture.md](docs/architecture.md). Full command reference:
[docs/usage.md](docs/usage.md).

## DICIE module (`src/docie/`)

Paper Fig. 1 path for **medical bills** and **salvage claims**: process →
classify → extract → aggregated response. Image-first (page OCR), with
optional FastAPI serving.

- Module README (CLI, Python API, REST, schema): [src/docie/README.md](src/docie/README.md)
- Design notes: [docs/docie_pipeline.md](docs/docie_pipeline.md)
- Taxonomies: `taxonomy/medical_bills.yaml`, `taxonomy/salvage_claims.yaml`
- Tests: `pytest tests/test_docie_pipeline.py`

This is separate from the markdown → memo chain in `src/pipeline/` (used by
the Discord bot). Prefer DICIE for paper-aligned workflow apps; prefer
`src.pipeline.orchestrator` when you need adjuster-style memos.

## Sample document corpus (`src/storage/`)

Queryable SQLite store for **synthetic** medical bills and salvage documentation
(Letters of Guarantee, salvage sales receipts, towing/storage attachments)
patterned after AmFam-style intake — without proprietary insurer data.

```bash
# Seed realistic samples + optional JSONL export
python -m src.storage seed --seed 42 --also-export

# Export for DICIE / classification / extraction training
python -m src.storage export --format docie --application salvage_claims \
  --out data/sample_corpus/exports/salvage_docie.jsonl
```

- Design notes: [docs/sample_document_corpus.md](docs/sample_document_corpus.md)
- Schemas: `data/schemas/medical_bill_skeleton.schema.json`,
  `data/schemas/salvage_document_skeleton.schema.json`
- Tests: `pytest tests/test_document_store.py tests/test_storage_training.py`
- Notebooks:
  - [`notebooks/sample_document_corpus_walkthrough.ipynb`](notebooks/sample_document_corpus_walkthrough.ipynb)
  - [`notebooks/sample_corpus_sql_integrations.ipynb`](notebooks/sample_corpus_sql_integrations.ipynb)
  - [`notebooks/sample_corpus_train_test_pipeline.ipynb`](notebooks/sample_corpus_train_test_pipeline.ipynb)

```bash
# optional: regenerate the three sample-corpus notebooks
python scripts/build_sample_corpus_notebooks.py
# interactive:
#   pip install -e ".[notebooks]"
#   jupyter notebook notebooks/sample_corpus_train_test_pipeline.ipynb
```

## Discord bot (Chloride)

Optional Discord front-end via [Chloride](https://github.com/S4IL21/chloride)
(Coral AI agent). Install the `discord` extra, set `DISCORD_TOKEN`, and run
`python -m src.discord_bot`.

The bot is an all-purpose server agent with an insurance-docs specialty:

- **Docs**: `/analyze`, `/analyze_url`, tool `analyze_insurance_document`
- **Notes / STT**: `/note`, `/transcribe`, `/remind`
- **DJ / vibes**: `/play`, `/queue`, `/vibe` (voice optional via `.[discord-voice]`)
- **Chat**: mention the bot or `--` prefix for free-form Chloride agent replies
- **Utils**: `/poll`, `/status`, `/help`, `/ping`

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

Score a completed run into the spreadsheet "Eval Results" summary:

```bash
python -m evaluation.metrics \
  --results evaluation/results/eval_run_2026-07-13/eval_results.jsonl \
  --output evaluation/results/eval_run_2026-07-13/summary.csv

# or score immediately after a live harness run
python -m evaluation.eval_harness ... --output-dir evaluation/results/eval_run_2026-07-13 --score
```

Classification → accuracy + macro F1; extraction → field micro-F1 (fuzzy fields
optional); memo generation → rubric coverage (LLM-judge scores merge in when
present on the JSONL rows).

### Cost model spreadsheet

Formula-driven workbook comparing frontier vs. local $/doc and monthly cost at
volume. Paste scores from `summary.csv` into the **Eval Results** sheet:

```bash
pip install -e ".[cost-model]"   # openpyxl
python -m evaluation.build_cost_model
# writes evaluation/cost_model/cost_model.xlsx
```

Sheets: Legend, Assumptions, Eval Results, Cost Per Doc, Scaling Projection,
Dashboard (with volume chart). Blue/yellow cells are editable inputs; green
cells are cross-sheet links; black cells are formulas.

## Experiment tracking (Weights & Biases)

Training, evaluation, and the seed generation pipeline log to [Weights & Biases](https://wandb.ai) by default:

- **Train**: Hugging Face Trainer metrics (`loss`, eval accuracy / F1), run config, `train_meta.json` artifacts (text DeBERTa, **ViT image**, and LayoutLMv3 extractors)
- **Eval**: summary metrics, per-class / field tables, confusion matrix (classifier + ViT), report + failure-mode artifacts
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

## Acknowledgments

Document taxonomy references publicly available ACORD form structures (acord.org). This project is not affiliated with or endorsed by ACORD.

Discord agent integration uses [Chloride](https://github.com/S4IL21/chloride) (fork of [Coral](https://github.com/uukelele/coral)), MIT licensed.
