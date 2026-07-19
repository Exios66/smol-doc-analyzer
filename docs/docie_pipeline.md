# DICIE Pipeline (Paper Fig. 1)

Implements the **Document Image Classification and Information Extraction**
chain from Raj, Dickinson & Fung, *Document Classification and Information
Extraction framework for Insurance Applications*.

```
Input (PDF / page images; structured · tabular · unstructured)
  → Stage 1  Document Processing
  → Stage 2  Document Classification  (+ page aggregation)
  → Stage 3  Information Extraction
  → Output   Aggregated prediction → response / downstream
```

This module (`src/docie/`) is **image-first** and application-scoped. It
complements the existing markdown → classify → extract → vision → summarize
chain in `src/pipeline/`, which remains the general ACORD intake / memo path.

## Applications (paper §III)

| Application | Classes | Extracted fields |
|-------------|---------|------------------|
| `medical_bills` | `hcfa`, `ub04`, `other` | claim_id, name, dob, patient_id, address |
| `salvage_claims` | `log`, `sales`, `other` | claim_id, vin, year, make, model |
| `acord` | ACORD intake taxonomy | claim / policy / loss fields |

Taxonomies live in `taxonomy/medical_bills.yaml` and `taxonomy/salvage_claims.yaml`.

## Stages

### 1. Document Processing (`src/docie/processing.py`)

- PDF → page images via PyMuPDF at configurable DPI (default **300**)
- Image preprocessing: grayscale conversion, retain dimensions
- OCR via PyTesseract when installed (`pip install -e ".[ocr]"`), with PDF
  text-layer fallback; produces word tokens + 0–1000 normalized boxes for
  LayoutLM-style extractors
- Plain text inputs are rendered to a page image so the image-first chain
  stays intact

### 2. Document Classification (`src/docie/classify.py`)

- Per-page scoring (keyword / alias heuristics; optional ViT when weights match)
- Document-level aggregation: confidence-weighted majority vote
- Business rule: prefer non-`other` on ties (paper medical / salvage setups)

### 3. Information Extraction (`src/docie/extract.py`)

- Field extraction conditioned on the predicted document type
- LayoutLM / token-classifier weights when present under `models/extractor*`
- Regex heuristics covering medical + salvage field patterns otherwise
- Missing expected fields (e.g. LOG without VIN) lower confidence and flag review

### 4. Output aggregation (`src/docie/aggregate.py`)

- Merge classification + extraction
- Route low-confidence / OCR-empty cases to human review
- Compact `response_payload()` for REST / claim-center downstream updates

## Usage

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

# Medical HCFA PDF / image
python -m src.docie --application medical_bills --pdf path/to/bill.pdf
python -m src.docie --application medical_bills --image path/to/page.png

# Batch JSONL
python -m src.docie \
  --application salvage_claims \
  --in tests/fixtures/sample_docie_documents.jsonl \
  --out data/pipeline/docie/salvage_demo.jsonl
```

Optional FastAPI server (paper §VI ECS/FastAPI serving shape):

```bash
pip install -e ".[serve]"
python -m src.docie.serve --application salvage_claims --port 8080
# POST /v1/predict  (multipart file)
# POST /v1/predict/text  (JSON {text, application, record_id})
```

## Relation to the chained analysis orchestrator

| Concern | `src/docie/` (this module) | `src/pipeline/` |
|---------|---------------------------|-----------------|
| Ordering | process → classify → extract → respond | markdown → classify → extract → vision → summarize |
| Primary signal | page images + OCR | structured markdown for LLM context |
| Applications | medical bills, salvage claims | ACORD intake + adjuster memo |
| Output | classification + fields (+ review flag) | classification + fields + memo |

Use DICIE when matching the paper flowchart / insurance workflow apps; use
the chained orchestrator when you need markdown-first LLM memos.
