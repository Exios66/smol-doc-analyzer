# Architecture — smol-doc-analyzer

## Pipeline

Two complementary inference chains are available:

### A. DICIE (paper Fig. 1) — `src/docie/`

Image-first chain matching Raj et al. *Document Classification and Information
Extraction framework for Insurance Applications*:

1. **Document Processing** — PDF / images → page images (300 DPI grayscale) + OCR
2. **Document Classification** — per-page classify → confidence-weighted aggregate
3. **Information Extraction** — LayoutLM / heuristic fields conditioned on class
4. **Output** — aggregate prediction, human-review flags, downstream response

Application profiles: `medical_bills`, `salvage_claims` (plus `acord`).

- Module README: [`src/docie/README.md`](../src/docie/README.md)
- Design notes: [`docs/docie_pipeline.md`](docie_pipeline.md)
- Taxonomies: `taxonomy/medical_bills.yaml`, `taxonomy/salvage_claims.yaml`

### B. Chained analysis (memo path) — `src/pipeline/`

Insurance document intake is split into specialized local components that run
as **one chained analysis action**. Stages are initiated in a fixed order and
execute chronologically; each stage reacts to prior stage outputs:

1. **To markdown** — PNG / PDF / text → compact structured markdown (headings + field tables) before any LLM call, to cut tokens and preserve layout cues
2. **Classification** — DeBERTa-v3 encoder (heuristic fallback) maps document text → taxonomy label; optionally a **ViT** image classifier maps rendered page images → the same taxonomy
3. **Extraction** — LayoutLMv3 / token classifier (heuristic fallback) pulls structured fields; conditioned on the predicted document type
4. **Vision LLM refine** — markdown-first local multimodal/text model (default target: Qwen2-VL class) corrects fields using classify+extract context; optional page image via `VISION_LLM_USE_IMAGE=1`
5. **Summarization** — generative LLM or template memo grounded in upstream markdown + payloads (not ground-truth skeletons)

Entry points:

```bash
# one document or JSONL — full chain in a single action
python -m src.pipeline.orchestrator --in data/synthetic/documents/documents.jsonl --out data/pipeline/analysis.jsonl --vision

# PNG or PDF directly (converted to markdown before LLM stages)
python -m src.pipeline.orchestrator --image scan.png --vision
python -m src.pipeline.orchestrator --pdf claim.pdf --vision

# batch with human-review queue for low-confidence cases
python -m src.pipeline.batch_runner --in data/synthetic/documents/documents.jsonl --out-dir data/pipeline/batch_run --vision
```

Upstream of training, an enhanced Phase 1 pipeline:

1. **Corpus ingest** — small public samples (FUNSD, DocLayNet, RVL-CDIP, legal writing, insurance distribution tables)
2. **Characteristic profiling** — layout, OCR noise, insurance histograms, legal vocabulary/reasoning style
3. **Synthetic generation** — claim skeletons → Stage A documents → Stage B memos → OCR noise variants

Legal corpora contribute **vocabulary and reasoning style only**. Classification labels remain insurance taxonomy categories in `taxonomy/acord_form_categories.yaml`.

## Repository map

```
src/
  generation/       # corpus ingest, profiling, skeleton/Stage A/B, noise
  classification/   # text DeBERTa + ViT image + TF-IDF Random Forest train/eval
  extraction/       # form render, LayoutLMv3 train/eval
  docie/            # paper Fig. 1 DICIE: process → classify → extract → respond
                    #   (see src/docie/README.md)
  storage/          # SQLite sample corpus for medical bills + salvage docs
  pipeline/         # orchestrator + markdown convert + batch_runner (memo chain)
  discord_bot/      # Chloride Discord front-end + analyze_insurance_document tool
  utils/            # config, provenance, LLM client, WandB tracking
discord/
  smol-doc-analyzer/  # Chloride bot workspace (config templates, compose)
data/
  schemas/          # claim_skeleton + medical_bill + salvage_document schemas
  profiles/         # committed characteristic profiles (small JSON)
  sample_corpus/    # queryable medical/salvage sample DB + seed exports
  raw/              # downloaded public samples (gitignored)
  synthetic/        # generated skeletons/documents/memos (gitignored)
  pipeline/         # inference outputs + markdown/render cache (gitignored)
taxonomy/           # ACORD + medical_bills + salvage_claims application labels
evaluation/         # frontier vs. local eval harness + reports/
  eval_harness.py   # Phase 7 cost/accuracy comparison runner
  metrics.py        # per-(task, backend) scoring → summary.csv
  build_cost_model.py  # builds evaluation/cost_model/cost_model.xlsx
  cost_model/       # committed spreadsheet template (formula-driven)
  pricing.yaml      # frontier $/M tokens + local GPU hourly rate
  prompts/          # versioned eval prompt templates
  reports/          # classification + extraction eval outputs
  results/          # harness JSONL/CSV run logs (gitignored)
```

## Discord (Chloride)

Optional Discord agent powered by [Chloride](https://github.com/S4IL21/chloride).
Install `pip install -e ".[discord]"`, set `DISCORD_TOKEN` (+ OpenRouter/AI key),
then `python -m src.discord_bot`.

**Slash commands** (synced on startup): `/analyze`, `/analyze_url`, `/status`, `/help`,
`/ping` — run the **memo chain** (`src/pipeline/`) from Discord's command picker.
DICIE (`src/docie/`) is exposed via CLI / optional FastAPI, not Discord slash commands.

**Chat / tools:** mention the bot or use the `--` prefix; the agent can call
`analyze_insurance_document`. Right-click → **Ask Me** analyzes a selected message.

See [discord/smol-doc-analyzer/README.md](../discord/smol-doc-analyzer/README.md).

## Data flow

```
Public corpora → profiles → skeletons → documents (+ noisy) → classifier (text and/or ViT on renders) / extractor
                                         └→ memos (Phase 4 training targets)

Paper Fig. 1 (DICIE):
  Inbound PDF/images → process (pages+OCR) → classify (aggregate) → extract → response / downstream

Memo chain:
  Inbound PNG/PDF/text → to_markdown → classify → extract → vision_llm → summarize
                                ↓
                       structured markdown (LLM context)
```

## Chronological reaction contract

`DocumentAnalysisOrchestrator` registers stages in initiation order and never
reorders by name. Each stage receives an accumulating `AnalysisContext`:

| Stage | Reads | Writes |
|-------|-------|--------|
| to_markdown | text / `image_path` / `pdf_path` | `markdown` (+ plain_text, token estimates) |
| classify | markdown plain_text (preferred) | `classification.document_type`, confidence |
| extract | markdown + classification | `extraction.fields*`, optional page render |
| vision_llm | markdown (+ optional image) + classify + extract | `vision.refined_fields` (merged into extraction) |
| summarize | markdown + all prior payloads | `summary.memo` |

Low-confidence stages append flags (`low_confidence_classification`, etc.)
rather than aborting the chain; `batch_runner` collects them into
`human_review_queue.jsonl`.

## Markdown conversion (token optimization)

`src/pipeline/markdown_convert.py` turns page content into compact markdown:

- **Text** → title / section headings + `| Field | Value |` tables for `Label: value` lines
- **PDF** → PyMuPDF (preferred) or pypdf text extract → same structuring
- **PNG** → optional `pytesseract` OCR (`pip install -e ".[ocr]"`), else fallback text when provided

Downstream LLM stages read this markdown instead of raw vision tokens by default.
Set `VISION_LLM_USE_IMAGE=1` only when you want an extra visual pass on high-RAM hosts.

## Local Vision LLM

On hosts with sufficient RAM/VRAM, set:

```bash
VISION_LLM_ENABLED=true
VISION_LLM_MODEL_PATH=/path/to/Qwen2-VL-2B-Instruct   # preferred
# or: VISION_LLM_LOAD=1 with VISION_LLM_MODEL=Qwen/Qwen2-VL-2B-Instruct
VISION_LLM_USE_IMAGE=0   # keep markdown-only LLM context (default)
```

Without local weights the vision stage still runs (heuristic refine over markdown)
so the single-action chain remains intact for development and CI.

## Design constraints

- No real insurer data in the repository
- Every synthetic record logged to `data/provenance_log.jsonl`
- Fixed held-out split (`data/synthetic/splits.json`) reused across Phases 2–5
- Template fallback for Stage A/B when OpenRouter is unavailable
- Inference chain falls back to heuristics when fine-tuned weights are absent
- LLM context prefers markdown over raw PNG/PDF pixels
- Training / eval / seed-pipeline runs tracked in Weights & Biases (`src/utils/wandb_utils.py`); disable with `--no-wandb` or `WANDB_MODE=disabled`
