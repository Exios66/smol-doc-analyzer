# Implementation & Development Plan — insurance-doc-ai

## Overview

This plan sequences the project into 7 phases. Each phase has a goal, concrete tasks, deliverables, and an exit criterion — the thing that must be true before moving on. Phases 1-3 can partially overlap; 4-6 are mostly sequential since each depends on the previous component's output.

---

## Phase 0: Setup & Scoping (est. 3-5 days)

**Goal:** Repository scaffolding, taxonomy v1, environment setup.

Tasks:
- Initialize repo with structure from `docs/architecture.md`
- Set up `pyproject.toml`, dependency management, `.env.example` for API keys (synthetic generation will call an LLM API)
- Draft `taxonomy/acord_form_categories.yaml` — initial classification schema based on public ACORD form index
- Draft `data/schemas/claim_skeleton.schema.json` — the structured intermediate representation
- Set up `data/provenance_log.jsonl` logging convention

Deliverable: working repo skeleton, taxonomy v1 draft.

Exit criterion: taxonomy has been sanity-checked against the public ACORD forms index and covers at minimum: applications, certificates/evidence, coverage-specific sections, loss/claims notices, supplements/endorsements.

---

## Phase 1: Synthetic Data Generation Pipeline (est. 2-3 weeks)

**Goal:** Produce a large, diverse, well-logged synthetic dataset of (document, memo) pairs.

Tasks:
- `skeleton_sampler.py`: build randomized claim skeleton generator, seeded with realistic value distributions (pull rough distributions from public Kaggle claims datasets for things like damage amounts, claim approval rates, loss types — not the raw data itself, just distributional shape)
- `stage_a_document_gen.py`: implement Stage A prompt (raw document text generation), batch-run against a subset of ACORD form types
- `stage_b_memo_gen.py`: implement Stage B prompt (memo generation conditioned on skeleton + Stage A output)
- `noise_injection.py`: build OCR-garble variants and multi-document claim bundles
- Build a small manual QA pass: sample 50-100 generated pairs, human-review for realism and label correctness
- Iterate on prompts based on QA findings (expect 2-3 rounds)

Deliverable: 5,000-10,000 synthetic (skeleton, document, memo) triples across all taxonomy categories, with provenance logging.

Exit criterion: manual QA pass shows >90% of sampled documents are plausible/internally consistent, and memo quality is judged "usable as training target" by manual review.

---

## Phase 2: Classification Model (est. 1-2 weeks)

**Goal:** Document-type classifier, smallest and fastest component — validates the taxonomy end to end.

Tasks:
- Prepare classification dataset from Stage A synthetic documents (document text → taxonomy label)
- Fine-tune DeBERTa-v3-base (or similar small encoder) on document type classification
- Build `eval.py`: accuracy, per-class F1, confusion matrix (watch for taxonomy categories that are hard to distinguish — likely candidates: certificate vs. evidence forms, similar coverage-section forms)
- Error analysis: check whether misclassifications reveal taxonomy design flaws (categories too similar, need merging)

Deliverable: trained classifier + eval report.

Exit criterion: >90% accuracy on held-out synthetic set; taxonomy revised if confusion analysis reveals structural issues (revisit Phase 0 taxonomy file if needed).

---

## Phase 3: Extraction Model (est. 3-4 weeks — budget the most time here)

**Goal:** Layout-aware field extraction from documents, including simulated scanned/OCR'd inputs.

Tasks:
- Convert synthetic documents into simulated "scanned form" representations (render as image with layout, or use OCR-style noisy text output) for training a layout-aware model
- Set up LayoutLMv3 or Donut as base model
- Define field extraction schema mapping model outputs back to `claim_skeleton.json` fields
- Fine-tune on synthetic (document → structured fields) pairs
- Build eval harness: field-level precision/recall, exact-match vs. partial-match scoring
- Stress-test against noisy/OCR-garbled variants from Phase 1
- Error analysis: identify which field types are hardest (dates, dollar amounts, free-text narrative fields will likely need different handling)

Deliverable: trained extraction model + eval report + documented failure modes.

Exit criterion: field-level extraction accuracy meets a defined threshold (propose ~85%+ on clean synthetic docs, lower tolerance acceptable on noisy variants) — and known failure modes are documented, not just measured.

---

## Phase 4: Summarization / Memo Generation (est. 2-3 weeks)

**Goal:** Fine-tune a small generative LLM to write adjuster-style memos from extracted claim data.

Tasks:
- Prepare LoRA fine-tuning dataset from Stage B synthetic memos, conditioned on Phase 3 extraction outputs (not the ground-truth skeleton — train on realistic upstream noise so the model learns to handle imperfect extraction)
- Select base model (Qwen2.5-7B-Instruct vs Llama-3.1-8B — run a quick 100-sample eval on both before committing)
- Configure and run LoRA fine-tuning
- Build eval harness: ROUGE/BERTScore against synthetic reference memos, plus a rubric-based LLM-judge eval (factual accuracy, completeness, tone) since surface-overlap metrics alone are weak for this task
- Human review pass on a sample of generated memos

Deliverable: fine-tuned LoRA adapter + eval report.

Exit criterion: LLM-judge rubric scores show generated memos are factually grounded in extracted data (no hallucinated claim details) and structurally match the target memo format.

---

## Phase 5: Pipeline Integration (est. 1-2 weeks)

**Goal:** Wire the three components into a single orchestrated pipeline.

Tasks:
- Build `orchestrator.py`: document in → classify → extract → summarize → memo out
- Build `batch_runner.py` for processing document batches
- Add error handling/fallback paths (e.g., low-confidence classification flags for human review rather than silent failure)
- End-to-end test on a held-out synthetic set spanning all taxonomy categories
- Include a Vision LLM refine stage in the same chronological chain (local VLM when RAM/VRAM allows; heuristic refine otherwise)

Deliverable: working end-to-end pipeline, runnable via CLI.

Exit criterion: pipeline processes a batch of held-out documents end to end without manual intervention, with confidence scores/flags surfaced for low-certainty cases.

**Status:** Implemented in `src/pipeline/` — stages execute in initiation order and each reacts to prior payloads. Inbound PNG/PDF/text is converted to structured markdown before LLM stages. Summarization uses template grounding until Phase 4 LoRA lands; Vision LLM loads only when `VISION_LLM_MODEL_PATH` / `VISION_LLM_LOAD` is configured (markdown-first by default).

---

## Phase 5b: DICIE paper path (Fig. 1) — status: implemented

**Goal:** Ship the Raj et al. image-first Document Image Classification and
Information Extraction chain alongside the memo orchestrator, scoped to the
paper's insurance applications.

Tasks:
- `src/docie/processing.py`: PDF/image → 300 DPI grayscale pages + OCR boxes
- `src/docie/classify.py`: per-page classify + confidence-weighted aggregate
- `src/docie/extract.py`: LayoutLM / heuristic fields conditioned on class
- `src/docie/aggregate.py` + `pipeline.py`: response payload + CLI batch runner
- Application taxonomies: `taxonomy/medical_bills.yaml`, `taxonomy/salvage_claims.yaml`
- Optional FastAPI serve path (`src/docie/serve.py`, `.[serve]` extra)
- Module + design docs: `src/docie/README.md`, `docs/docie_pipeline.md`

Deliverable: runnable DICIE CLI / Python API / optional REST for medical bills
and salvage claims, with human-review routing.

Exit criterion: fixture suite (`tests/test_docie_pipeline.py`) passes end to end
with heuristic backends (no trained weights required).

**Status:** Implemented in `src/docie/`. Complements Phase 5 memo chain; Discord
`/analyze` continues to call `src/pipeline/`.

---

## Phase 6: Deployment Packaging (est. 1-2 weeks)

**Goal:** Package for offline/in-house deployment.

Tasks:
- Quantize summarization model (GGUF conversion for llama.cpp, or vLLM serving config)
- Write `deployment/hardware_sizing.md`: throughput/latency/cost estimates at different hardware tiers (single GPU, CPU-only, etc.)
- Build Docker container for full pipeline (memo chain + optional DICIE REST via `src.docie.serve`)
- Document setup/run instructions for someone with no ML background to deploy this internally
- Package DICIE FastAPI serving shape (paper §VI) for ECS / container deploy when needed

Deliverable: deployable package + hardware sizing documentation.

Exit criterion: pipeline runs successfully in a clean Docker environment from a documented setup process.

---

## Phase 7: Benchmarking & Documentation (est. 1 week, ongoing)

**Goal:** Cost/accuracy story and final documentation for both audiences (portfolio + AmFam handoff).

Tasks:
- Build `evaluation/eval_harness.py` (alias: `evaluation/benchmarks.py`): compare pipeline accuracy and cost-per-document against a frontier-model API baseline on the same held-out set
- Write `docs/architecture.md` final version (design rationale; dual chains: DICIE + memo)
- Keep `src/docie/README.md` + `docs/docie_pipeline.md` aligned with the Fig. 1 path
- Write `docs/data_provenance.md` (full synthetic data sourcing disclosure, including DICIE fixtures)
- Write `docs/handoff/amfam_pipeline_recipe.md` — the consulting deliverable, pointing only at reproducible pipeline/training code, not synthetic data specifics
- Write portfolio-facing project writeup (results, architecture diagram, cost comparison)

Deliverable: complete documentation set, benchmark report, public-facing writeup.

Exit criterion: both deliverables (AmFam recipe doc, GitHub portfolio writeup) are complete and internally consistent with the data provenance disclosure.

---

## Cross-cutting concerns (apply throughout, not a separate phase)

- **Provenance logging**: every synthetic data generation run must log to `data/provenance_log.jsonl` from Phase 1 onward — do not defer this.
- **Taxonomy stability**: changes to `taxonomy/acord_form_categories.yaml` after Phase 2 require re-running classification eval; track taxonomy versions.
- **Eval-set discipline**: hold out a fixed synthetic eval set from the start of Phase 1 and never train on it — reuse the same set across Phases 2-5 for comparable metrics.
- **Cost tracking**: log synthetic generation API costs from day one — this becomes part of your own cost-comparison narrative in Phase 7.

## Effort notes

Calendar-week estimates above are planning heuristics for human schedules, not
agent runtime. Difficulty concentrates in Phase 3 (layout-aware extraction) and
Phase 4 (memo LoRA); Phase 5b (DICIE) is largely complete with heuristic
fallback backends.
