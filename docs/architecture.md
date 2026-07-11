# Architecture — smol-doc-analyzer

## Pipeline

Insurance document intake is split into specialized local components that run
as **one chained analysis action**. Stages are initiated in a fixed order and
execute chronologically; each stage reacts to prior stage outputs:

1. **Classification** — DeBERTa-v3 encoder (heuristic fallback) maps document text → taxonomy label
2. **Extraction** — LayoutLMv3 / token classifier (heuristic fallback) pulls structured fields; conditioned on the predicted document type
3. **Vision LLM refine** — optional local multimodal model (default target: Qwen2-VL class) reads the page image and corrects fields using classify+extract context; heuristic refine when no VLM weights are loaded
4. **Summarization** — generative LLM or template memo grounded only in upstream payloads (not ground-truth skeletons)

Entry points:

```bash
# one document or JSONL — full chain in a single action
python -m src.pipeline.orchestrator --in data/synthetic/documents/documents.jsonl --out data/pipeline/analysis.jsonl --vision

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
  classification/   # dataset prep, DeBERTa train/eval
  extraction/       # form render, LayoutLMv3 train/eval
  pipeline/         # orchestrator + batch_runner (Phase 5 chained inference)
  utils/            # config, provenance, LLM client
data/
  schemas/          # claim_skeleton.schema.json
  profiles/         # committed characteristic profiles (small JSON)
  raw/              # downloaded public samples (gitignored)
  synthetic/        # generated skeletons/documents/memos (gitignored)
  pipeline/         # inference outputs + render cache (gitignored)
taxonomy/           # ACORD-inspired document type labels
evaluation/reports/ # classification + extraction eval outputs
```

## Data flow

```
Public corpora → profiles → skeletons → documents (+ noisy) → classifier / extractor
                                         └→ memos (Phase 4 training targets)

Inbound document → classify → extract → vision_llm → summarize → memo + flags
```

## Chronological reaction contract

`DocumentAnalysisOrchestrator` registers stages in initiation order and never
reorders by name. Each stage receives an accumulating `AnalysisContext`:

| Stage | Reads | Writes |
|-------|-------|--------|
| classify | `document.text` | `classification.document_type`, confidence |
| extract | text + classification | `extraction.fields*`, optional page render |
| vision_llm | image + classify + extract | `vision.refined_fields` (merged into extraction) |
| summarize | all prior payloads | `summary.memo` |

Low-confidence stages append flags (`low_confidence_classification`, etc.)
rather than aborting the chain; `batch_runner` collects them into
`human_review_queue.jsonl`.

## Local Vision LLM

On hosts with sufficient RAM/VRAM, set:

```bash
VISION_LLM_ENABLED=true
VISION_LLM_MODEL_PATH=/path/to/Qwen2-VL-2B-Instruct   # preferred
# or: VISION_LLM_LOAD=1 with VISION_LLM_MODEL=Qwen/Qwen2-VL-2B-Instruct
```

Without local weights the vision stage still runs (heuristic refine) so the
single-action chain remains intact for development and CI.

## Design constraints

- No real insurer data in the repository
- Every synthetic record logged to `data/provenance_log.jsonl`
- Fixed held-out split (`data/synthetic/splits.json`) reused across Phases 2–5
- Template fallback for Stage A/B when OpenRouter is unavailable
- Inference chain falls back to heuristics when fine-tuned weights are absent
