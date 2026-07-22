# Usage Guide — smol-doc-analyzer

**Version:** `1.0.0-beta` (`1.0.0b0`)

Complete reference for installing, configuring, generating data, training,
running both inference pipelines, managing the sample corpus, evaluating,
serving, and operating the Discord bot.

| Doc | Purpose |
|-----|---------|
| [architecture.md](architecture.md) | Dual-pipeline design and repository map |
| [docie_pipeline.md](docie_pipeline.md) | Paper Fig. 1 DICIE path |
| [sample_document_corpus.md](sample_document_corpus.md) | SQLite medical / salvage sample store |
| [data_provenance.md](data_provenance.md) | Synthetic-only data disclosure |
| [CHANGELOG.md](../CHANGELOG.md) | Version history through `1.0.0-beta` |

---

## 1. Install

```bash
# core + test/lint tooling (+ openpyxl for the cost spreadsheet)
pip install -e ".[dev]"

# optional extras
pip install -e ".[ocr]"          # pytesseract OCR for page images
pip install -e ".[serve]"        # FastAPI DICIE REST server
pip install -e ".[notebooks]"    # Jupyter + RF / pipeline notebooks
pip install -e ".[cost-model]"   # openpyxl only (also included in .[dev])
pip install -e ".[discord]"      # Chloride Discord bot
pip install -e ".[discord,discord-voice]"  # + voice DJ (needs ffmpeg)
```

Python **3.11+** required (`requires-python` in `pyproject.toml`).

---

## 2. Secrets & configuration

```bash
python scripts/setup_env.py          # create gitignored .env from template
# edit .env — paste keys (never commit)
python scripts/setup_env.py --status # which secrets are set (values never printed)
```

| Variable | Used by |
|----------|---------|
| `OPENROUTER_API_KEY` | Synthetic LLM generation, frontier eval, Discord AI bridge |
| `WANDB_API_KEY` | Training / eval / generation experiment tracking |
| `HF_TOKEN` | Optional Hub downloads |
| `DISCORD_TOKEN` | Interactive Chloride bot |
| `DISCORD_WEBHOOK_URL` | One-way analysis notifications |
| `DISCORD_AI_API_KEY` | Bot LLM key override (defaults to OpenRouter) |
| `OPENAI_API_KEY` | Preferred Whisper backend for `/transcribe` |
| `VISION_LLM_*` / `SUMMARIZER_*` | Local multimodal / generative stages |
| `SAMPLE_CORPUS_*` | SQLite sample corpus paths |
| Path overrides (`*_OUTPUT_DIR`, `MODELS_DIR`, …) | See `.env.example` |

Config loads from environment + `.env` via `src.utils.config.Config.load()`.
Placeholder secret values are treated as unset.

When paid OpenRouter credits are exhausted (HTTP 402), generation auto-routes
to free models (`OPENROUTER_FREE_FALLBACK_MODELS` / `OPENROUTER_PREFER_FREE`).

---

## 3. Two inference pipelines

| Pipeline | Entry | Stages | Best for |
|----------|-------|--------|----------|
| **DICIE (paper Fig. 1)** | `python -m src.docie` | process → classify → extract → respond | Medical bills / salvage claims; image-first OCR |
| **Memo chain** | `python -m src.pipeline.orchestrator` | markdown → classify → extract → vision → summarize | ACORD intake + adjuster memo |

### 3.1 DICIE — document processing → classification → extraction

```bash
# Salvage Letter of Guarantee (text demo → rendered page image)
python -m src.docie \
  --application salvage_claims \
  --text "LETTER OF GUARANTEE
Claim Number: CLM-2024-100200
VIN: 1HGCM82633A004352
Year: 2018
Make: Honda
Model: Accord" \
  --response-only

# Medical / salvage files
python -m src.docie --application medical_bills --pdf path/to/bill.pdf
python -m src.docie --application salvage_claims --image path/to/log.png

# Batch JSONL
python -m src.docie \
  --application salvage_claims \
  --in tests/fixtures/sample_docie_documents.jsonl \
  --out data/pipeline/docie/salvage_demo.jsonl

# Compact downstream payload only
python -m src.docie --application medical_bills --pdf bill.pdf --response-only

# Skip OCR (use PDF text layer / provided text)
python -m src.docie --application salvage_claims --pdf doc.pdf --no-ocr
```

**Applications:** `salvage_claims` · `medical_bills` · `acord`  
Taxonomies: `taxonomy/salvage_claims.yaml`, `taxonomy/medical_bills.yaml`,
`taxonomy/acord_form_categories.yaml`.

Module README (CLI flags, prediction schema, REST contract):
[src/docie/README.md](../src/docie/README.md).

**REST server (optional):**

```bash
pip install -e ".[serve]"
python -m src.docie.serve --application salvage_claims --port 8080
# GET  /health
# POST /v1/predict          multipart file (PDF/image)
# POST /v1/predict/text     JSON {text, application, record_id, response_only}
```

Python API: `from src.docie import process_document, DociePipeline`.

### 3.2 Memo chain — markdown → classify → extract → vision → summarize

```bash
# JSONL batch
python -m src.pipeline.orchestrator \
  --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl \
  --out data/pipeline/analysis.jsonl \
  --vision --limit 20

# Single inputs
python -m src.pipeline.orchestrator --image path/to/scan.png --vision
python -m src.pipeline.orchestrator --pdf path/to/claim.pdf --vision
python -m src.pipeline.orchestrator --vision --text "LOSS NOTICE\nClaim Number: CLM-1\n..."

# Batch + human-review queue
python -m src.pipeline.batch_runner \
  --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl \
  --out-dir data/pipeline/batch_demo \
  --vision --limit 20
```

Flags: `--vision` / `--no-vision`, `--classifier-dir`, `--extractor-dir`.

Vision LLM (optional local weights):

```bash
VISION_LLM_ENABLED=true
VISION_LLM_MODEL_PATH=/path/to/Qwen2-VL-2B-Instruct
# or VISION_LLM_LOAD=1 with VISION_LLM_MODEL=...
VISION_LLM_USE_IMAGE=0   # markdown-only context (default)
```

Python API: `from src.pipeline import analyze_document, DocumentAnalysisOrchestrator`.

The Discord bot `/analyze` commands call this memo chain (not DICIE).

---

## 4. Sample document corpus (SQLite)

Queryable house for synthetic medical bills and salvage documentation
(Letters of Guarantee, salvage sales, towing/storage). See
[sample_document_corpus.md](sample_document_corpus.md).

```bash
# Seed realistic samples + optional JSONL export
python -m src.storage seed --seed 42 --also-export

# Inspect
python -m src.storage summary
python -m src.storage list --application salvage_claims
python -m src.storage show sal-log-001

# Export for DICIE / classification / extraction
python -m src.storage export --format docie --application salvage_claims \
  --out data/sample_corpus/exports/salvage_docie.jsonl

python -m src.storage export --format classification \
  --out data/sample_corpus/exports/all_classification.jsonl

# Import existing gold / fixtures
python -m src.storage import-jsonl --in data/eval/docie_eval_set.jsonl
python -m src.storage import-jsonl --in tests/fixtures/sample_docie_documents.jsonl

# Run DICIE against an export
python -m src.docie \
  --application salvage_claims \
  --in data/sample_corpus/exports/salvage_docie.jsonl \
  --out data/pipeline/docie/salvage_from_corpus.jsonl
```

Default DB: `data/sample_corpus/documents.db` (gitignored; regenerable).

---

## 5. Synthetic data generation

```bash
# Refresh characteristic profiles + small seed run
python -m src.generation.run_seed_pipeline --n 240
# optional Hub ingest: add --ingest

# or step-by-step:
python -m src.generation.corpus_ingest
python -m src.generation.characteristic_profiler
python -m src.generation.skeleton_sampler --n 5000 --out data/synthetic/skeletons/
python -m src.generation.stage_a_document_gen --in data/synthetic/skeletons/skeletons_n5000_seed42.jsonl
python -m src.generation.stage_b_memo_gen --in data/synthetic/documents/documents_from_skeletons_n5000_seed42.jsonl
python -m src.generation.noise_injection --in data/synthetic/documents/documents_from_skeletons_n5000_seed42.jsonl
```

Without `OPENROUTER_API_KEY`, Stage A/B use **template** renderers. Every synthetic
record is logged to `data/provenance_log.jsonl`.

Profiles live in `data/profiles/` (`insurance_distributions.json`, layout, OCR,
legal style, document surface). Claim schema: `data/schemas/claim_skeleton.schema.json`.

---

## 6. Classification training & eval

### 6.1 Text classifier (DeBERTa / DistilBERT smoke)

```bash
python -m src.classification.prepare_dataset \
  --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl

python -m src.classification.train_classifier \
  --prepared data/synthetic/documents/classification_prepared --smoke

python -m src.classification.eval \
  --model-dir models/classifier_smoke \
  --prepared data/synthetic/documents/classification_prepared
```

Omit `--smoke` for full DeBERTa-v3 training on GPU.

### 6.2 ViT image classifier

```bash
python -m src.extraction.render_forms \
  --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl \
  --out data/synthetic/documents/rendered

python -m src.classification.prepare_image_dataset \
  --in data/synthetic/documents/rendered/rendered.jsonl

python -m src.classification.train_vit_classifier \
  --prepared data/synthetic/documents/rendered/vit_classification_prepared --smoke

python -m src.classification.eval_vit \
  --model-dir models/vit_classifier_smoke \
  --prepared data/synthetic/documents/rendered/vit_classification_prepared
```

### 6.3 TF-IDF + Random Forest baseline

```bash
python -m src.classification.train_random_forest
# or interactively:
#   pip install -e ".[notebooks]"
#   jupyter notebook notebooks/random_forest_text_handwriting_classification.ipynb
```

Reports land in `evaluation/reports/` (`classification_report.*`,
`vit_classification_report.*`, `random_forest_classification_report.md`).

WandB flags on train/eval CLIs: `--wandb` / `--no-wandb` / `--wandb-project` /
`--wandb-run-name`. Global off: `WANDB_MODE=disabled`.

---

## 7. Extraction training & eval

```bash
python -m src.extraction.render_forms \
  --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl \
  --out data/synthetic/documents/rendered

python -m src.extraction.prepare_dataset \
  --in data/synthetic/documents/rendered/rendered.jsonl

python -m src.extraction.train_extractor \
  --prepared data/synthetic/documents/rendered/extraction_prepared --smoke

python -m src.extraction.eval \
  --model-dir models/extractor_smoke \
  --prepared data/synthetic/documents/rendered/extraction_prepared
```

Full LayoutLMv3: omit `--smoke`. Noisy stress: prepare/eval against
`rendered_noisy` / `extraction_prepared_noisy` variants.

---

## 8. Evaluation harness (frontier vs local)

```bash
# Plan only (no API spend)
python -m evaluation.eval_harness \
  --eval-set data/eval/eval_set.jsonl \
  --tasks classification extraction memo_generation \
  --backends anthropic openai local \
  --n-samples 50 \
  --output-dir evaluation/results/eval_run_demo \
  --dry-run

# Live run
python -m evaluation.eval_harness \
  --eval-set data/eval/eval_set.jsonl \
  --backends anthropic openai local \
  --output-dir evaluation/results/eval_run_demo

# Score → summary.csv
python -m evaluation.metrics \
  --results evaluation/results/eval_run_demo/eval_results.jsonl \
  --output evaluation/results/eval_run_demo/summary.csv

# Or: add --score to the harness invocation
```

DICIE-specific metrics (paper Table I / II style):

```bash
python -m src.docie.eval --all
python -m src.docie.eval --application salvage_claims
# → evaluation/reports/docie_{application}_{metrics.json,report.md}
```

Cost spreadsheet:

```bash
pip install -e ".[cost-model]"   # or .[dev]
python -m evaluation.build_cost_model
# → evaluation/cost_model/cost_model.xlsx
```

Pricing: `evaluation/pricing.yaml`. Prompt templates: `evaluation/prompts/`.

---

## 9. Discord bot (Chloride)

```bash
pip install -e ".[discord]"
python scripts/setup_env.py   # set DISCORD_TOKEN (+ OPENROUTER_API_KEY)
cp discord/smol-doc-analyzer/config.yaml.example discord/smol-doc-analyzer/config.yaml
python -m src.discord_bot
```

| Capability | Commands / tools |
|------------|------------------|
| Docs | `/analyze`, `/analyze_url`, tool `analyze_insurance_document` |
| Notes / STT | `/note`, `/transcribe`, `/remind` |
| DJ / vibes | `/play`, `/queue`, `/skip`, `/stop`, `/join`, `/leave`, `/vibe` |
| Chat | mention bot or `--` prefix |
| Utils | `/poll`, `/status`, `/help`, `/ping` |

Details: [discord/smol-doc-analyzer/README.md](../discord/smol-doc-analyzer/README.md).

Outbound webhook (no bot process):

```bash
python -m src.discord_bot.webhook --check
python -m src.discord_bot.webhook --text "LOSS NOTICE\nClaim Number: CLM-1\n..."
python -m src.discord_bot.webhook --pdf path/to/claim.pdf
```

macOS autostart: `./scripts/install_discord_bot_loginitem.sh` or
`./scripts/install_discord_bot_launchagent.sh`.

---

## 10. Testing

```bash
pip install -e ".[dev,ocr,serve]"
pytest                         # full suite
pytest tests/test_docie_pipeline.py tests/test_pipeline.py -q
pytest tests/test_document_store.py tests/test_random_forest.py -q
ruff check src tests
```

| Suite | Covers |
|-------|--------|
| `test_docie_pipeline.py` / `test_docie_eval.py` | Fig. 1 DICIE + metrics |
| `test_pipeline.py` | Memo chain orchestrator + markdown |
| `test_document_store.py` | SQLite sample corpus |
| `test_vit_prepare.py` / `test_random_forest.py` | Image prep & RF |
| `test_generation.py` | Skeletons, templates, noise |
| `test_eval_harness.py` / `test_eval_metrics.py` / `test_cost_model.py` | Eval stack |
| `test_llm_client.py` / `test_config_provenance.py` / `test_wandb_utils.py` | Utils |
| `test_discord_bot.py` / `test_discord_commands.py` | Discord helpers |
| `test_bugfix_regressions.py` | Locked-in audit fixes |

Fixtures: `tests/fixtures/sample_documents.jsonl`, `sample_docie_documents.jsonl`,
`sample_skeletons.jsonl`. Demo data: `data/notebook_demo/`.

---

## 11. Repository map (commands ↔ modules)

```
src/generation/       corpus_ingest, characteristic_profiler, skeleton_sampler,
                      stage_a_document_gen, stage_b_memo_gen, noise_injection,
                      run_seed_pipeline
src/classification/   prepare_dataset, prepare_image_dataset, train_*, eval*,
                      random_forest
src/extraction/       render_forms, prepare_dataset, train_extractor, eval
src/docie/            Fig. 1 pipeline + serve + eval
src/pipeline/         memo-chain orchestrator, markdown_convert, batch_runner
src/storage/          SQLite sample corpus (seed / export / import)
src/discord_bot/      Chloride runner, slash commands, tools, notes, vibes, STT
src/utils/            config, io, llm_client, prompts, provenance, wandb_utils
evaluation/           eval_harness, metrics, local_backends, build_cost_model
taxonomy/             acord + medical_bills + salvage_claims
data/                 schemas, profiles, sample_corpus, synthetic (gitignored bulk)
notebooks/            pipeline + DICIE + RF walkthroughs
scripts/              setup_env.py, Discord autostart, build_rf_notebook.py
discord/              Chloride workspace (config templates, compose)
```

---

## 12. Typical end-to-end workflows

**A. Local salvage / medical intake (paper path)**

1. Install `.[dev,ocr]` (and `.[serve]` if exposing HTTP).
2. `python -m src.docie --application salvage_claims --pdf inbound.pdf --response-only`
3. Route `needs_human_review` rows from batch `*.human_review.jsonl`.

**B. Grow a queryable sample house, then train / run DICIE**

1. `python -m src.storage seed --seed 42 --also-export`
2. Export `--format docie` / `classification` / `extraction` as needed.
3. Train or run `python -m src.docie --in …`.

**C. Train classifiers/extractors on synthetic data, then memo-analyze**

1. `python -m src.generation.run_seed_pipeline --n 240`
2. Prepare + train classifier / extractor (smoke or full).
3. `python -m src.pipeline.orchestrator --in … --vision --out data/pipeline/analysis.jsonl`

**D. Cost / accuracy bake-off**

1. Build or refresh `data/eval/eval_set.jsonl`.
2. Dry-run then live `evaluation.eval_harness`.
3. `evaluation.metrics` → paste into cost model spreadsheet.

**E. Discord-assisted review**

1. Configure token + OpenRouter key; run `python -m src.discord_bot`.
2. `/analyze` attachments or paste text; `/status` to confirm secrets (no values leaked).
