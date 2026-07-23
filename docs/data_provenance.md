---
title: "Data Provenance"
subtitle: "Synthetic-only disclosure — no real insurer data"
---

::: {.callout-important}
## No real insurance company data
Training and evaluation data are either **public corpora** (distributional /
stylistic characteristics only) or **synthetically generated** from randomized
claim skeletons.
:::

[Usage](usage.md) · [Changelog](CHANGELOG.md) · [Sample corpus](sample_document_corpus.md)

## Public sources (prime examples)

| Source | What we take | What we do **not** take |
|--------|--------------|-------------------------|
| FUNSD (form understanding) | Field label lexicon, key-value layout patterns, OCR noise shape | Raw forms as training labels for our taxonomy |
| DocLayNet (layout) | Layout class frequencies; legal/regulatory prose texture from `laws_and_regulations` pages | DocLayNet category labels as classifier targets |
| RVL-CDIP (document images) | Surface characteristics of form/letter/memo/invoice/email classes for style conditioning | RVL-CDIP class labels as our taxonomy |
| Public insurance claim tables | Histograms for loss amounts, loss types, state mix (shape only) | Individual real claim rows as documents |
| Legal writing samples (e.g. public opinions / pile-of-law style text) | Vocabulary, discourse markers, IRAC-ish reasoning templates | Legal document types as classification labels |

Legal text is injected only into narrative sections of insurance documents (claims correspondence, supporting evidence, memo reasoning). The classifier label set remains insurance-only.

## Synthetic generation

1. **Skeleton sampling** — randomized `ClaimSkeleton` objects validated against `data/schemas/claim_skeleton.schema.json`, seeded by `data/profiles/insurance_distributions.json`
2. **Stage A** — document text (LLM via OpenRouter when configured, else deterministic templates conditioned on profiles)
3. **Stage B** — adjuster memo text from skeleton + Stage A
4. **Noise injection** — OCR garble using `data/profiles/ocr_noise_profile.json`

Every record is appended to `data/provenance_log.jsonl` with stage, source, model (if any), and prompt/profile version.

## DICIE application fixtures

The paper-aligned DICIE path (`src/docie/`) uses the same synthetic-only rule.
Committed examples in `tests/fixtures/sample_docie_documents.jsonl` are
hand-written fictional HCFA / UB-04 / LOG / sales texts for CI — not real
claimant documents. Application label sets live in
`taxonomy/medical_bills.yaml` and `taxonomy/salvage_claims.yaml`. Batch DICIE
runs append provenance rows with `stage=docie_pipeline`.

## Sample medical + salvage document corpus

The queryable corpus store (`src/storage/`, see
[sample_document_corpus.md](sample_document_corpus.md)) houses larger sets of
fictional medical bills and salvage documentation (Letters of Guarantee, salvage
sales receipts, towing/storage attachments) patterned after AmFam-style intake
surfaces. No proprietary insurer files are ingested. Seed/import events log
`stage=sample_corpus_seed` / `sample_corpus_import`.

Skeleton schemas:

- `data/schemas/medical_bill_skeleton.schema.json`
- `data/schemas/salvage_document_skeleton.schema.json`

## RVL-CDIP public index

The queryable RVL-CDIP SQL store (`src/rvl_cdip/`, see
[rvl_cdip_sql.md](rvl_cdip_sql.md)) indexes the public
[aharley/rvl_cdip](https://huggingface.co/datasets/aharley/rvl_cdip) label
lists. Hub downloads and the SQLite DB are confined to `.venv/rvl_cdip/`
(covered by the existing `.venv/` gitignore). Build events log
`stage=rvl_cdip_build`. The ~38 GB image archive is never fetched unless
explicitly opted in.

## Committed vs gitignored

- **Committed:** schemas, taxonomy (ACORD + medical bills + salvage claims), characteristic profiles (`data/profiles/*.json`), tiny test fixtures (including DICIE samples), sample-corpus seed exports under `data/sample_corpus/seeds/`
- **Gitignored:** bulk `data/raw/*`, synthetic JSONL outputs, provenance log, trained model weights, pipeline/DICIE caches under `data/pipeline/`, sample corpus SQLite DB + regenerable exports under `data/sample_corpus/`, RVL-CDIP artifacts under `.venv/rvl_cdip/`

## Reproducibility

Characteristic profiles are versioned JSON committed to the repo so synthetic generation can run without re-downloading multi-GB corpora. Corpus ingest scripts remain available to refresh profiles from Hub samples when needed.

::: {.see-also}
### See also

[Sample Document Corpus](sample_document_corpus.md) ·
[RVL-CDIP SQL Index](rvl_cdip_sql.md) ·
[Architecture](architecture.qmd) · [About](about.qmd) ·
[Bugfix audit](bugfix_audit_round2.md)
:::
