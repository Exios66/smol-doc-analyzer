---
title: "RVL-CDIP SQL Index"
subtitle: "Queryable public document-image labels under .venv"
---

Queryable SQLite index for the public
[aharley/rvl_cdip](https://huggingface.co/datasets/aharley/rvl_cdip) dataset
(400,000 grayscale document images across 16 classes).

::: {.callout-important}
## Downloads stay in `.venv`
All Hub downloads, HF caches, and the SQLite database are confined to
`.venv/rvl_cdip/`. Nothing is written under `data/` or `~/.cache/huggingface`.
:::

::: {.callout-warning}
## Large archive is opt-in
The image tarball (`rvl-cdip.tar.gz`) is **~38 GB**. The default `build`
path downloads only the split label files (~17 MB) and indexes all 400k
rows from those. Image download requires
`--i-understand-large-download` and a free-space preflight check.
:::

## Layout

| Path | Role |
|------|------|
| `.venv/rvl_cdip/hf_home/` | Forced `HF_HOME` / Hub cache |
| `.venv/rvl_cdip/source/labels/` | `train.txt` / `test.txt` / `val.txt` |
| `.venv/rvl_cdip/source/rvl-cdip.tar.gz` | Optional image archive |
| `.venv/rvl_cdip/rvl_cdip.db` | Queryable SQLite index |

## Schema

| Table | Role |
|-------|------|
| `labels` | 16 class ids → names |
| `documents` | One row per image (`split`, `label_id`, `image_relpath`, optional `image_abspath`) |
| `download_events` | Audit trail of Hub fetches |
| `schema_meta` | Schema version + build metadata |

## Quickstart

```bash
# Label files → SQL index (safe default; no 38 GB download)
python -m src.rvl_cdip build

# Inspect
python -m src.rvl_cdip summary
python -m src.rvl_cdip labels
python -m src.rvl_cdip list --split train --label invoice --limit 5
python -m src.rvl_cdip paths

# Ad-hoc SQL (SELECT only)
python -m src.rvl_cdip query \
  "SELECT l.name AS label, COUNT(*) AS n
   FROM documents d JOIN labels l ON l.label_id = d.label_id
   GROUP BY l.name ORDER BY n DESC"

# Optional: pull the image archive into .venv (explicit opt-in)
python -m src.rvl_cdip download-images --i-understand-large-download
```

## Class labels

| id | name | id | name |
|---:|------|---:|------|
| 0 | letter | 8 | file folder |
| 1 | form | 9 | news article |
| 2 | email | 10 | budget |
| 3 | handwritten | 11 | invoice |
| 4 | advertisement | 12 | presentation |
| 5 | scientific report | 13 | questionnaire |
| 6 | scientific publication | 14 | resume |
| 7 | specification | 15 | memo |

## Module

- `src/rvl_cdip/paths.py` — `.venv` layout + HF cache env pinning
- `src/rvl_cdip/download.py` — cautious Hub fetch (labels default; images opt-in)
- `src/rvl_cdip/store.py` — SQLite store + read-only `query()`
- `src/rvl_cdip/__main__.py` — CLI

## Recreation sampling notebook

For a seeded draw of **60–70 documents from each class (0–15)** plus SQL
showcase queries, run
[`notebooks/rvl_cdip_recreation_sampling.ipynb`](notebooks/rvl_cdip_recreation_sampling.ipynb)
(regenerate with `python scripts/build_rvl_cdip_recreation_notebook.py`).
Exports land under `data/notebook_demo/rvl_cdip_recreation/`.

Related: the lighter streaming sample ingest used for characteristic profiles
remains in `src/generation/corpus_ingest.py` (`ingest_rvl_cdip`). This module
is the full queryable SQL house for the public dataset.
