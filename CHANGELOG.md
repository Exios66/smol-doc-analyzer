# Changelog

All notable changes to **smol-doc-analyzer** (SmallDocAnalyzer) are documented
in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with [PEP 440](https://peps.python.org/pep-0440/) package versions
(`1.0.0-beta` ↔ `1.0.0b0`).

Versions below `1.0.0-beta` record the incremental build-up of the repository
from scaffolding through the current beta baseline.

---

## [1.0.0-beta] — 2026-07-22

### Added

- `docs/usage.md` — end-to-end usage guide covering install, both inference
  pipelines, sample corpus, training, eval, Discord, and tests
- `CHANGELOG.md` — versioned history of prior iterations through this beta

### Changed

- Package version set to **`1.0.0b0`** (`1.0.0-beta`) as the current release
  baseline
- README, architecture, plan, and related docs aligned to the live repository
  layout (DICIE, memo chain, sample corpus, Discord, eval harness)
- `.[cost-model]` optional extra restored as an alias for the cost spreadsheet
  dependency (`openpyxl`; also included in `.[dev]`)

### Status

Phases **0–3**, **5** (memo chain), **5b** (DICIE), sample corpus store,
Discord bot, eval/cost harness, and classical/ViT classifiers are in place.
Phase **4** (summarizer LoRA) remains pending — memos use template grounding
with an optional local generative hook.

---

## [0.7.0] — 2026-07-22

### Added

- SQLite **sample document corpus** (`src/storage/`) for synthetic medical
  bills and salvage documentation (Letters of Guarantee, salvage sales,
  towing/storage attachments)
- CLI: `python -m src.storage seed|summary|list|show|export|import-jsonl`
- Schemas: `medical_bill_skeleton.schema.json`,
  `salvage_document_skeleton.schema.json`
- Docs: `docs/sample_document_corpus.md`; provenance notes for corpus seed/import
- Tests: `tests/test_document_store.py`

---

## [0.6.0] — 2026-07-19

### Added

- Full DICIE module README (`src/docie/README.md`) and aligned project docs
  (`docs/docie_pipeline.md`, architecture / Discord cross-links)
- OpenRouter **free-model fallback** when paid credits are exhausted
  (`OPENROUTER_FREE_FALLBACK_MODELS`, `OPENROUTER_PREFER_FREE`)

### Changed

- Random Forest classification notebook and logging polish
- Evaluation datasets and document descriptions refreshed
- Discord bot setup documentation expanded

---

## [0.5.0] — 2026-07-15 – 2026-07-19

### Added

- Paper-aligned **DICIE pipeline** (`src/docie/`): process → classify →
  extract → aggregated response for `medical_bills` and `salvage_claims`
- Application taxonomies: `taxonomy/medical_bills.yaml`,
  `taxonomy/salvage_claims.yaml`
- Optional FastAPI serve path (`python -m src.docie.serve`, `.[serve]`)
- DICIE eval reports under `evaluation/reports/docie_*`
- Expanded Discord slash commands and agent tools (notes, STT, DJ/vibes, polls)

### Fixed

- Audit bugfixes across pipeline, Discord, extraction, and eval (#20)

### Changed

- Package version officially bumped from `0.1.0` → `0.5.0` in-tree
- README feature surface updated for DICIE + Discord + RF

---

## [0.4.0] — 2026-07-11 – 2026-07-14

### Added

- **TF-IDF + Random Forest** classifier for typed text and handwriting/OCR
  noise (`src/classification/random_forest.py`, train CLI, notebook)
- Frontier vs. local **evaluation harness** (`evaluation/eval_harness.py`,
  `metrics.py`, `local_backends.py`, `pricing.yaml`)
- Formula-driven **cost model** spreadsheet builder
  (`evaluation/build_cost_model.py` → `evaluation/cost_model/cost_model.xlsx`)
- Versioned eval prompt templates under `evaluation/prompts/`

---

## [0.3.0] — 2026-07-11

### Added

- **Chloride Discord AI** front-end (`src/discord_bot/`) wired to the memo
  analysis chain
- Slash commands for document analysis (`/analyze`, `/analyze_url`) plus
  webhook outbound notifications
- Discord workspace under `discord/smol-doc-analyzer/` (config templates,
  Docker Compose)
- macOS Login Item / LaunchAgent install scripts under `scripts/`
- Minimal Discord intents for Chloride bot login

---

## [0.2.0] — 2026-07-11

### Added

- Phase **5 chained document analysis** orchestrator (`src/pipeline/`):
  to_markdown → classify → extract → vision_llm → summarize
- PNG/PDF → structured markdown conversion before LLM stages
- Batch runner with human-review queue for low-confidence cases
- **Weights & Biases** experiment tracking for train / eval / seed generation
- **ViT** document-image classification path (prepare, train, eval)
- Local secrets setup via `scripts/setup_env.py` and gitignored `.env`

---

## [0.1.0] — 2026-07-10 – 2026-07-11

### Added

- Repository scaffolding, `pyproject.toml`, taxonomy, claim skeleton schema
- Characteristic profiles from public document / layout / legal-style priors
- Synthetic pipeline: skeleton → Stage A documents → Stage B memos → OCR noise
  (OpenRouter LLM with template fallback)
- Document-type classifier train/eval (DeBERTa / DistilBERT smoke)
- Field extraction train/eval (LayoutLMv3 / smoke) with noisy stress reporting
- Provenance logging to `data/provenance_log.jsonl`
- Initial README, architecture plan, and data provenance disclosure
- No real insurer data — public priors + synthetic generation only

---

## Version map

| Version | Milestone |
|---------|-----------|
| `0.1.0` | Phases 0–3: synthetic data, classify, extract |
| `0.2.0` | Phase 5 memo chain, WandB, ViT, local secrets |
| `0.3.0` | Discord / Chloride integration |
| `0.4.0` | Random Forest + frontier/local eval + cost model |
| `0.5.0` | DICIE (Fig. 1) + Discord expansions + audit fixes |
| `0.6.0` | DICIE docs, OpenRouter free fallback, RF polish |
| `0.7.0` | SQLite sample medical / salvage corpus |
| `1.0.0-beta` | Current beta baseline + usage guide + changelog |

[1.0.0-beta]: https://github.com/Exios66/smol-doc-analyzer/releases/tag/v1.0.0-beta
[0.7.0]: https://github.com/Exios66/smol-doc-analyzer/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Exios66/smol-doc-analyzer/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Exios66/smol-doc-analyzer/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Exios66/smol-doc-analyzer/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Exios66/smol-doc-analyzer/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Exios66/smol-doc-analyzer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Exios66/smol-doc-analyzer/releases/tag/v0.1.0
