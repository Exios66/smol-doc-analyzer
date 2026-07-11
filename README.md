# smol-doc-analyzer

A small, locally-deployable LLM pipeline for insurance document classification, extraction, and memo generation — built as a cost-efficient alternative to frontier enterprise models for high-volume document intake.

## Why this exists

Insurance operations (claims intake, underwriting submissions, policy servicing) generate huge volumes of structured-but-messy documents — ACORD forms, loss notices, adjuster reports, correspondence. Processing these with frontier API models (GPT-4/Claude-class) works, but at volume it's expensive and requires sending documents to a third party. This project explores whether a small, specialized, **locally-hosted** pipeline can match "good enough" accuracy at a fraction of the cost, fully in-house.

Rather than one large model trying to do everything, this pipeline splits the problem into three specialized components:

1. **Classification** — a small fine-tuned encoder (DeBERTa-v3 class) identifies document type and category
2. **Extraction** — a layout-aware model (LayoutLMv3/Donut class) pulls structured fields from forms, including scanned/OCR'd documents
3. **Summarization** — a small fine-tuned generative LLM (7-8B, quantized) writes adjuster-style memos and analysis from the extracted content

All three run offline, on modest hardware, with no per-token API costs.

## Status

Early development. Synthetic data generation and taxonomy definition in progress. No trained models yet.

## Data disclosure

This project uses **no real insurance company data**. All training data is either:
- Publicly available (ACORD form templates, public document-layout datasets, public Kaggle insurance datasets used for realistic field distributions), or
- Synthetically generated (fictional claims, documents, and memos produced via LLM generation from randomized skeletons)

Every synthetic record is logged in `data/provenance_log.jsonl` with its generation source. See `docs/data_provenance.md` for full detail.

This project began as an exploratory build related to a document-processing problem discussed with American Family Insurance in a consulting context. No AmFam data, documents, or proprietary information were used in this repository. See `docs/handoff/amfam_pipeline_recipe.md` for the separate internal deliverable design (pipeline/recipe only, meant to be run by AmFam on their own data).

## Quickstart (once implemented)

```bash
# install
pip install -e .

# generate synthetic training data
python -m src.generation.skeleton_sampler --n 5000 --out data/synthetic/skeletons/
python -m src.generation.stage_a_document_gen --in data/synthetic/skeletons/ --out data/synthetic/documents/
python -m src.generation.stage_b_memo_gen --in data/synthetic/documents/ --out data/synthetic/memos/

# train components
python -m src.classification.train_classifier
python -m src.extraction.train_extractor
python -m src.summarization.train_lora

# run full pipeline on a document
python -m src.pipeline.orchestrator --input path/to/document.pdf
```

## Repository structure

See `docs/architecture.md` for the full design rationale and repository map.

## Evaluation

Benchmarks compare this pipeline's accuracy and cost-per-document against a frontier-model API baseline. See `evaluation/benchmarks.py` and `evaluation/reports/`.

## License

TBD — add before making repository public.

## Acknowledgments

Document taxonomy references publicly available ACORD form structures (acord.org). This project is not affiliated with or endorsed by ACORD.
