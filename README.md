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
- Document-type classifier train/eval
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

# refresh profiles (optional Hub ingest: add --ingest)
python -m src.generation.run_seed_pipeline --n 240

# prepare + train classifier (smoke uses DistilBERT on CPU)
python -m src.classification.prepare_dataset --in data/synthetic/documents/documents_from_skeletons_n240_seed42.jsonl
python -m src.classification.train_classifier --prepared data/synthetic/documents/classification_prepared --smoke
python -m src.classification.eval --model-dir models/classifier_smoke --prepared data/synthetic/documents/classification_prepared

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

# full-scale generation (when ready)
python -m src.generation.skeleton_sampler --n 5000 --out data/synthetic/skeletons/
python -m src.generation.stage_a_document_gen --in data/synthetic/skeletons/skeletons_n5000_seed42.jsonl
python -m src.generation.stage_b_memo_gen --in data/synthetic/documents/documents_from_skeletons_n5000_seed42.jsonl
```

For GPU training, omit `--smoke` and use the default DeBERTa-v3 / LayoutLMv3 model names.

On high-RAM local hosts, point `VISION_LLM_MODEL_PATH` at a downloaded Qwen2-VL
(or similar) checkpoint and set `VISION_LLM_LOAD=1` to refine extraction from
page images inside the same chain.

## Repository structure

See [docs/architecture.md](docs/architecture.md).

## Evaluation

Reports land in `evaluation/reports/` (`classification_report.*`, `extraction_report.json`, `failure_modes.md`).

## License

TBD — add before making repository public.

## Acknowledgments

Document taxonomy references publicly available ACORD form structures (acord.org). This project is not affiliated with or endorsed by ACORD.
