# Architecture — smol-doc-analyzer

## Pipeline

Insurance document intake is split into three specialized local components:

1. **Classification** — DeBERTa-v3 encoder maps document text → taxonomy label; optionally a **ViT** image classifier maps rendered page images → the same taxonomy (Kaggle/HF-style document image classification)
2. **Extraction** — LayoutLMv3 pulls structured fields from rendered/OCR'd forms
3. **Summarization** (Phase 4+) — small generative LLM writes adjuster-style memos

Upstream of training, an enhanced Phase 1 pipeline:

1. **Corpus ingest** — small public samples (FUNSD, DocLayNet, RVL-CDIP, legal writing, insurance distribution tables)
2. **Characteristic profiling** — layout, OCR noise, insurance histograms, legal vocabulary/reasoning style
3. **Synthetic generation** — claim skeletons → Stage A documents → Stage B memos → OCR noise variants

Legal corpora contribute **vocabulary and reasoning style only**. Classification labels remain insurance taxonomy categories in `taxonomy/acord_form_categories.yaml`.

## Repository map

```
src/
  generation/       # corpus ingest, profiling, skeleton/Stage A/B, noise
  classification/   # text DeBERTa + ViT image train/eval
  extraction/       # form render, LayoutLMv3 train/eval
  pipeline/         # orchestrator (Phase 5)
  utils/            # config, provenance, LLM client, WandB tracking
data/
  schemas/          # claim_skeleton.schema.json
  profiles/         # committed characteristic profiles (small JSON)
  raw/              # downloaded public samples (gitignored)
  synthetic/        # generated skeletons/documents/memos (gitignored)
taxonomy/           # ACORD-inspired document type labels
evaluation/reports/ # classification + extraction eval outputs
```

## Data flow

```
Public corpora → profiles → skeletons → documents (+ noisy) → classifier (text and/or ViT on renders) / extractor
                                         └→ memos (Phase 4 training targets)
```

## Design constraints

- No real insurer data in the repository
- Every synthetic record logged to `data/provenance_log.jsonl`
- Fixed held-out split (`data/synthetic/splits.json`) reused across Phases 2–5
- Template fallback for Stage A/B when OpenRouter is unavailable
- Training / eval / seed-pipeline runs tracked in Weights & Biases (`src/utils/wandb_utils.py`); disable with `--no-wandb` or `WANDB_MODE=disabled`
