# `src/docie` — DICIE pipeline

Paper-aligned **Document Image Classification and Information Extraction**
(DICIE) chain from Raj, Dickinson & Fung, *Document Classification and
Information Extraction framework for Insurance Applications* (Fig. 1).

```
Input (PDF / page images / text)
  → Stage 1  Document Processing
  → Stage 2  Document Classification  (+ page aggregation)
  → Stage 3  Information Extraction
  → Output   Aggregated prediction → response / downstream
```

This module is **image-first** and **application-scoped** (medical bills,
salvage claims, optional ACORD). It complements the markdown → classify →
extract → vision → summarize memo chain in [`src/pipeline/`](../pipeline/).

Longer design notes: [`docs/docie_pipeline.md`](../../docs/docie_pipeline.md).
Architecture overview: [`docs/architecture.md`](../../docs/architecture.md).

---

## Install

```bash
# from repo root
pip install -e ".[dev]"

# optional: OCR for scanned PDFs / images
pip install -e ".[ocr]"          # requires system tesseract

# optional: FastAPI REST server (paper §VI)
pip install -e ".[serve]"
```

On Debian/Ubuntu, OCR also needs:

```bash
sudo apt-get install -y tesseract-ocr
```

---

## Quickstart

```bash
# Salvage Letter of Guarantee (text demo)
python -m src.docie \
  --application salvage_claims \
  --text "LETTER OF GUARANTEE
Claim Number: CLM-2024-100200
VIN: 1HGCM82633A004352
Year: 2018
Make: Honda
Model: Accord" \
  --response-only

# Medical bill PDF / page image
python -m src.docie --application medical_bills --pdf path/to/bill.pdf
python -m src.docie --application salvage_claims --image path/to/log.png

# Batch JSONL → prediction JSONL + summary + human-review queue
python -m src.docie \
  --application salvage_claims \
  --in tests/fixtures/sample_docie_documents.jsonl \
  --out data/pipeline/docie/salvage_demo.jsonl
```

Optional REST server:

```bash
pip install -e ".[serve]"
python -m src.docie.serve --application salvage_claims --port 8080
# GET  /health
# POST /v1/predict       (multipart file: PDF or image)
# POST /v1/predict/text  (JSON {text, application, record_id})
```

---

## Module layout

| File | Role |
|------|------|
| `__init__.py` | Public exports: `DociePipeline`, `process_document`, `DociePrediction`, `ProcessedDocument` |
| `__main__.py` | `python -m src.docie` entry |
| `pipeline.py` | End-to-end orchestrator + CLI (`DociePipeline`, `process_document`, `run_file`) |
| `processing.py` | Stage 1 — PDF/image → page images (300 DPI grayscale) + OCR |
| `classify.py` | Stage 2 — per-page classify + confidence-weighted aggregate |
| `extract.py` | Stage 3 — LayoutLM / heuristic field extraction |
| `aggregate.py` | Output stage — merge results, review flags, downstream payload |
| `applications.py` | Load application profiles from `taxonomy/*.yaml` |
| `types.py` | Dataclasses: pages, classification, extraction, prediction |
| `serve.py` | Optional FastAPI server (`python -m src.docie.serve`) |

---

## Applications

Configured via taxonomy YAML under `taxonomy/`:

| Application | Taxonomy | Classes | Extracted fields |
|-------------|----------|---------|------------------|
| `medical_bills` | `taxonomy/medical_bills.yaml` | `hcfa`, `ub04`, `other` | claim_id, name, dob, patient_id, address |
| `salvage_claims` | `taxonomy/salvage_claims.yaml` | `log`, `sales`, `other` | claim_id, vin, year, make, model |
| `acord` | `taxonomy/acord_form_categories.yaml` | ACORD intake labels | claim / policy / loss fields |

Business rules (prefer non-`other` on ties, review confidence threshold) live
in each taxonomy's `business_rules` block. Load programmatically:

```python
from src.docie.applications import list_applications, load_application

list_applications()  # ['acord', 'medical_bills', 'salvage_claims']
profile = load_application("salvage_claims")
profile.labels            # ['log', 'sales', 'other']
profile.extraction_fields # ['claim_id', 'vin', 'year', 'make', 'model']
```

---

## Stages

### 1. Document Processing (`processing.py`)

- PDF → page images via PyMuPDF at configurable DPI (default **300**)
- Image preprocessing: grayscale conversion, retain native dimensions
- OCR via PyTesseract when installed (`.[ocr]`), with PDF text-layer fallback
- Word tokens + 0–1000 normalized boxes for LayoutLM-style extractors
- Plain text inputs are rendered to a page image so the image-first chain stays intact
- Page cache under `data/pipeline/cache/docie/<application>/pages/`

### 2. Document Classification (`classify.py`)

- Per-page scoring (keyword / alias heuristics; optional ViT when `--vit-model-dir` matches)
- Document-level aggregation: confidence-weighted majority vote
- Prefer non-`other` on ties when `prefer_non_other` is set (medical / salvage)

### 3. Information Extraction (`extract.py`)

- Fields conditioned on the predicted document type
- LayoutLM / token-classifier weights when present under `models/extractor*`
- Regex heuristics covering medical + salvage field patterns otherwise
- Missing expected fields (e.g. LOG without VIN) lower confidence and flag review

### 4. Output aggregation (`aggregate.py`)

- Merge classification + extraction
- Route low-confidence / OCR-empty / missing-field cases to human review
- Compact `response_payload()` for REST / claim-center downstream updates
- Optional `downstream_sink` callback on `DociePipeline`

---

## CLI reference

```text
python -m src.docie [OPTIONS]

  -a, --application   medical_bills | salvage_claims | acord
                      (default: salvage_claims)
  --in PATH           Input JSONL of documents
  --out PATH          Output JSONL (default under data/pipeline/docie/)
  --text TEXT         Analyze a single text blob
  --image PATH        Document page image (PNG/JPEG/…)
  --pdf PATH          Multi/single-page PDF
  --record-id ID      Record id for ad-hoc runs (default: adhoc)
  --limit N           Cap JSONL rows processed
  --dpi N             PDF render DPI (default: 300)
  --no-ocr            Skip pytesseract OCR
  --response-only     Print compact downstream payload
  --vit-model-dir DIR Optional ViT classifier weights
  --extractor-dir DIR Optional LayoutLM extractor weights
```

Provide **one** of `--in`, `--text`, `--image`, or `--pdf`.

Batch runs also write:

- `<out>.summary.json` — counts, confidences, label histogram
- `<out.stem>.human_review.jsonl` — compact payloads needing review
- A provenance row at `data/provenance_log.jsonl` (`stage=docie_pipeline`)

### Input JSONL row shape

```json
{
  "record_id": "sal-log-001",
  "text": "LETTER OF GUARANTEE\n...",
  "pdf_path": null,
  "image_path": null,
  "source_path": null
}
```

Any of `text`, `pdf_path`, `image_path`, or `source_path` / `path` / `file_path`
may be set. Extra keys are kept as `metadata`.

Fixture samples: [`tests/fixtures/sample_docie_documents.jsonl`](../../tests/fixtures/sample_docie_documents.jsonl).

---

## Python API

```python
from src.docie import DociePipeline, process_document

# Convenience: one document → prediction dict
result = process_document(
    application="salvage_claims",
    text="LETTER OF GUARANTEE\nClaim Number: CLM-1\nVIN: 1HGCM82633A004352\n...",
)
print(result["document_type"], result["fields"])

# Full control
pipe = DociePipeline(
    application="medical_bills",
    dpi=300,
    run_ocr=True,
    vit_model_dir=None,       # optional Path to ViT weights
    extractor_dir=None,       # defaults to models/extractor or extractor_smoke
)
prediction = pipe.process(
    record_id="med-001",
    pdf_path="path/to/bill.pdf",
)
print(prediction.classification.label)
print(prediction.extraction.fields_flat)
print(prediction.needs_human_review)
print(prediction.response_payload())  # compact REST shape

# Batch
from pathlib import Path
from src.docie.pipeline import run_file

run_file(
    Path("tests/fixtures/sample_docie_documents.jsonl"),
    Path("data/pipeline/docie/out.jsonl"),
    application="salvage_claims",
)
```

### Prediction schema (full `to_dict()`)

```json
{
  "record_id": "sal-log-001",
  "application": "salvage_claims",
  "document_type": "log",
  "fields": {"claim_id": "CLM-...", "vin": "...", "year": "2018", "make": "Honda", "model": "Accord"},
  "classification": {
    "label": "log",
    "confidence": 0.85,
    "backend": "heuristic_text",
    "aggregation": "majority_vote",
    "page_predictions": [],
    "flags": []
  },
  "extraction": {
    "fields": {"claim_id": ["CLM-..."], "vin": ["..."]},
    "fields_flat": {"claim_id": "CLM-...", "vin": "..."},
    "backend": "heuristic_regex",
    "document_type": "log",
    "confidence": 0.9,
    "flags": []
  },
  "processing": {
    "source_kind": "text",
    "n_pages": 1,
    "dpi": 300,
    "ocr_backends": ["rendered_text"]
  },
  "flags": [],
  "needs_human_review": false,
  "stage_timings_ms": {
    "document_processing": 12.3,
    "document_classification": 1.1,
    "information_extraction": 0.8,
    "output_aggregation": 0.2
  }
}
```

### Compact response (`response_payload()` / `--response-only`)

```json
{
  "record_id": "sal-log-001",
  "application": "salvage_claims",
  "document_type": "log",
  "classification_confidence": 0.85,
  "fields": {"claim_id": "CLM-...", "vin": "..."},
  "extraction_confidence": 0.9,
  "needs_human_review": false,
  "flags": []
}
```

---

## REST API (`serve.py`)

```bash
python -m src.docie.serve --application salvage_claims --host 0.0.0.0 --port 8080
```

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `GET` | `/health` | — | Status + known applications |
| `POST` | `/v1/predict` | multipart: `file`, optional `application_name`, `record_id`, `response_only` | PDF or image upload |
| `POST` | `/v1/predict/text` | JSON: `{text, application?, record_id?, response_only?}` | Text-only demo path |

Default responses use the compact payload (`response_only=true`).

---

## Relation to `src/pipeline/`

| Concern | `src/docie/` (this module) | `src/pipeline/` |
|---------|---------------------------|-----------------|
| Ordering | process → classify → extract → respond | markdown → classify → extract → vision → summarize |
| Primary signal | page images + OCR | structured markdown for LLM context |
| Applications | medical bills, salvage claims | ACORD intake + adjuster memo |
| Output | classification + fields (+ review flag) | classification + fields + memo |
| Discord bot | not wired (use CLI / REST) | `/analyze`, `analyze_insurance_document` |

Use DICIE when matching the paper flowchart / workflow apps; use the chained
orchestrator when you need markdown-first LLM memos.

---

## Testing

```bash
pytest tests/test_docie_pipeline.py -q
```

Coverage includes application loading, heuristic classify/extract, aggregation
business rules, end-to-end `DociePipeline` / `run_file`, and review routing.
Fixtures live in `tests/fixtures/sample_docie_documents.jsonl` (synthetic only —
no real insurer data).

---

## Dependencies & fallbacks

| Capability | Requirement | Fallback when missing |
|------------|-------------|------------------------|
| PDF render | `pymupdf` (core dep) | — |
| OCR | `.[ocr]` + system tesseract | PDF text layer / empty OCR (flags review) |
| ViT classify | `--vit-model-dir` with trained weights | keyword / alias heuristics |
| LayoutLM extract | `models/extractor` or `extractor_smoke` | regex field heuristics |
| REST serve | `.[serve]` | CLI / Python API only |

Heuristic paths keep CI and local demos working without trained weights or OCR.
