---
title: "Sample Document Corpus"
subtitle: "Queryable synthetic medical / salvage store"
---

Queryable storage for **synthetic** medical-bill and salvage-claim documents
used for analysis, evaluation, and model fine-tuning. Command cheatsheet:
[Usage §4](usage.md) · [Commands](reference/commands.qmd).

::: {.callout-important}
## No proprietary claim files
This project does **not** use real American Family (or any insurer) claim
files. The corpus houses **reality-tied, fictional** examples. See
[Data Provenance](data_provenance.md).
:::

Surfaces covered:

- Casualty medical billing review (HCFA / CMS-1500, UB-04, non-standard statements)
- Total-loss salvage workflows (Letters of Guarantee, salvage sales receipts,
  towing/storage and related attachments)

## Why this exists

DICIE fixtures and eval JSONL (`tests/fixtures/sample_docie_documents.jsonl`,
`data/eval/docie_eval_set.jsonl`) are excellent for CI, but they are not a
queryable house for growing sample corpora, claim bundles, ground-truth fields,
or train/val/test splits. The sample corpus store fills that gap.

## Schema (SQLite)

Default DB path: `data/sample_corpus/documents.db` (gitignored; regenerable).

| Table | Role |
|-------|------|
| `claims` | Claim-level container (carrier, policy, loss metadata) |
| `documents` | Canonical document text + skeleton JSON + split |
| `document_fields` | Ground-truth / extracted / annotation fields |
| `document_pages` | Optional page image / OCR attachments |
| `provenance_events` | In-DB seed/import/export audit trail |
| `schema_meta` | Schema version |

JSON schemas for structured skeletons:

- `data/schemas/medical_bill_skeleton.schema.json`
- `data/schemas/salvage_document_skeleton.schema.json`

These are richer than the DICIE extraction field sets in
`taxonomy/medical_bills.yaml` / `taxonomy/salvage_claims.yaml` — they also
capture providers, lienholders, payoff amounts, sales tax, diagnosis/procedure
codes, etc., while export still projects the taxonomy fields for training.

## Quickstart

```bash
# Seed a diverse synthetic corpus (canonical CI fixtures + generated samples)
python -m src.storage seed --seed 42 --also-export

# Inspect
python -m src.storage summary
python -m src.storage list --application salvage_claims
python -m src.storage show sal-log-001

# Export for DICIE / classification / extraction training
python -m src.storage export --format docie \
  --application medical_bills \
  --out data/sample_corpus/exports/medical_docie.jsonl

python -m src.storage export --format classification \
  --out data/sample_corpus/exports/all_classification.jsonl

python -m src.storage export --format extraction \
  --application salvage_claims \
  --out data/sample_corpus/exports/salvage_extraction.jsonl

# Import existing DICIE gold / fixtures
python -m src.storage import-jsonl --in data/eval/docie_eval_set.jsonl
python -m src.storage import-jsonl --in tests/fixtures/sample_docie_documents.jsonl
```

Run DICIE against an export:

```bash
python -m src.storage export --format docie --application salvage_claims \
  --out data/sample_corpus/exports/salvage_docie.jsonl

python -m src.docie \
  --application salvage_claims \
  --in data/sample_corpus/exports/salvage_docie.jsonl \
  --out data/pipeline/docie/salvage_from_corpus.jsonl
```

## Document types

### Medical bills (`application=medical_bills`)

| Type | Description |
|------|-------------|
| `hcfa` | CMS-1500 / HCFA physician claim with carrier, patient, DX/CPT |
| `ub04` | UB-04 institutional bill with type of bill + revenue codes |
| `other` | Non-standard clinic / urgent-care statements |

Ground-truth export fields (taxonomy): `claim_id`, `name`, `dob`, `patient_id`, `address`.

### Salvage claims (`application=salvage_claims`)

| Type | Description |
|------|-------------|
| `log` | Bank / lender **Letter of Guarantee** for lien payoff |
| `sales` | Salvage sales receipt / bill of sale |
| `other` | Towing, storage, and related salvage attachments |

Ground-truth export fields (taxonomy): `claim_id`, `vin`, `year`, `make`, `model`.

Claim **bundles** group LOG + sales + towing docs (or HCFA + UB-04 + statement)
under one `claim_id` so multi-document salvage/medical files can be analyzed
together.

## Module layout

| Path | Role |
|------|------|
| `src/storage/store.py` | SQLite `DocumentStore` (CRUD, import/export) |
| `src/storage/schema.py` | DDL + schema version |
| `src/storage/types.py` | `ClaimRecord`, `DocumentRecord`, `FieldRecord` |
| `src/storage/sample_generator.py` | Realistic synthetic medical + salvage templates |
| `src/storage/training.py` | Prepare train/val/test + TF-IDF RF from the store |
| `src/storage/__main__.py` | CLI (`seed`, `summary`, `list`, `show`, `export`, `import-jsonl`) |

## Notebooks

| Notebook | Focus |
|----------|--------|
| [Corpus walkthrough](notebooks/sample_document_corpus_walkthrough.ipynb) | Generate → seed → export → DICIE |
| [SQL integrations](notebooks/sample_corpus_sql_integrations.ipynb) | DDL, CRUD, joins, provenance, analytics |
| [Train → test pipeline](notebooks/sample_corpus_train_test_pipeline.ipynb) | SQL → prepare → train → test → DICIE eval |

Regenerate the notebooks from the builder script:

```bash
python scripts/build_sample_corpus_notebooks.py
```

## Provenance

- Every seed/import writes an in-DB `provenance_events` row.
- Seed/import also append to `data/provenance_log.jsonl` with
  `stage=sample_corpus_seed` / `sample_corpus_import`.
- All generated records set `is_synthetic=1` and
  `metadata.carrier_style=american_family_simulation`.

::: {.see-also}
### See also
[Pipeline hub](pipelines/index.qmd) · [DICIE](docie_pipeline.md) ·
[Notebooks portal](notebooks/index.qmd) · [Data Provenance](data_provenance.md)
:::
