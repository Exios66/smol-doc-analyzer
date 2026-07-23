---
title: "Bugfix audit (round 2)"
subtitle: "Regression notes after PR #20"
---

::: {.callout-note}
## Related
[Changelog](CHANGELOG.md) · [Architecture](architecture.md) ·
[Home](index.qmd)
:::

Follow-up to PR #20 (`Fix full-repo audit bugs`). This pass covers regressions
and new silent bugs introduced after that merge (Discord SSRF hardening side
effects, DICIE serve, sample corpus store, RF splits, eval free-routing).

## Issues → fixes

| Issue | Severity | Symptom | Fix |
|------|----------|---------|-----|
| [#28](https://github.com/Exios66/smol-doc-analyzer/issues/28) | Critical | `/analyze` PDF/PNG attachments always fail | Pass inbox files via `local_path=` instead of `file_url` |
| [#29](https://github.com/Exios66/smol-doc-analyzer/issues/29) | High | SSRF via HTTP redirects to private IPs | Disable auto-redirects; re-validate every hop; size cap |
| [#30](https://github.com/Exios66/smol-doc-analyzer/issues/30) | High | DICIE upload path traversal via `record_id` | Sanitize `record_id`; resolve path under temp dir |
| [#31](https://github.com/Exios66/smol-doc-analyzer/issues/31) | High | Upsert keeps stale gold fields | Replace field set per role on non-empty `fields` |
| [#32](https://github.com/Exios66/smol-doc-analyzer/issues/32) | High | RF typed/noisy split leakage (+ crash) | Split by unique `record_id`; object dtype + fallback |
| [#33](https://github.com/Exios66/smol-doc-analyzer/issues/33) | Medium | `Carrier Name:` extracted as patient | Line-leading `Name:` / `Patient:` only |
| [#34](https://github.com/Exios66/smol-doc-analyzer/issues/34) | Medium | PNG collisions for `::` vs `__` IDs | Hash-suffixed `_cache_safe_id` in `render_forms` |
| [#35](https://github.com/Exios66/smol-doc-analyzer/issues/35) | Medium | Eval cost wrong after free fallback | Persist used model; `$0` when `is_free_model` |

## Code touchpoints

- `src/discord_bot/commands.py`, `src/discord_bot/tools.py` — #28, #29
- `src/docie/serve.py` — #30
- `src/storage/store.py` — #31
- `src/classification/random_forest.py` — #32
- `src/docie/extract.py` — #33
- `src/extraction/render_forms.py` — #34
- `evaluation/eval_harness.py`, `src/utils/llm_client.py` — #35
- `tests/test_bugfix_regressions.py` — regression coverage for #28–#35

## Verification

```bash
pytest tests/test_bugfix_regressions.py tests/test_discord_bot.py \
  tests/test_document_store.py tests/test_random_forest.py \
  tests/test_docie_pipeline.py tests/test_eval_harness.py -q
```

## Out of scope / already fixed in PR #20

Silent OCR confidence, VLM decode, ACORD substring FPs, Discord private-IP /
`file://` host checks (initial), BIO/bbox LayoutLM issues, OCR noise banding,
skeleton dates/splits, eval latency-on-failure, memo judge `correct`, etc.
