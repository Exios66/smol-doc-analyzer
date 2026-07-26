#!/usr/bin/env python3
"""Generate sample-corpus walkthrough notebooks under notebooks/."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "notebooks"

SETUP = r'''
from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

CWD = Path.cwd().resolve()
# Walk up from docs/notebooks/ (or notebooks/) until pyproject.toml is found.
REPO_ROOT = next(
    (p for p in (CWD, *CWD.parents) if (p / "pyproject.toml").exists()),
    None,
)
assert REPO_ROOT is not None, f"Could not find repo root from {CWD}"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.docie import DociePipeline
from src.docie.applications import list_applications, load_application
from src.docie.eval import evaluate_application
from src.docie.pipeline import run_file
from src.storage import DocumentStore
from src.storage.sample_generator import generate_claim_bundle, generate_corpus
from src.storage.schema import DDL, SCHEMA_VERSION
from src.storage.training import (
    fit_tfidf_random_forest,
    prepare_both_applications,
)
from src.storage.types import ClaimRecord, DocumentRecord, FieldRecord
from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
cfg = Config.load()

DEMO = REPO_ROOT / "data" / "notebook_demo" / "sample_corpus"
DEMO.mkdir(parents=True, exist_ok=True)
DB_PATH = DEMO / "documents.db"
EXPORTS = DEMO / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)
PREPARED = DEMO / "prepared"
MODELS = DEMO / "models"
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
print(f"repo:     {REPO_ROOT}")
print(f"demo db:  {DB_PATH}")
print(f"exports:  {EXPORTS}")
print(f"schema v: {SCHEMA_VERSION}")
print(f"apps:     {list_applications()}")
'''.strip("\n")


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str):
    return nbf.v4.new_code_cell(source.strip("\n"))


def write_nb(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = cells
    out = OUT_DIR / name
    out.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {out}")
    return out


def build_walkthrough() -> Path:
    cells = [
        md(
            """
# Sample Document Corpus Walkthrough — `src/storage/`

End-to-end showcase of the **synthetic medical-bill + salvage-claim corpus** added for
analysis and fine-tuning without proprietary American Family Insurance data.

This notebook covers the new pipelines:

1. Rich skeleton schemas (medical + salvage)
2. Realistic sample generation (HCFA / UB-04 / LOG / sales / towing)
3. SQLite store seeding
4. Claim bundles (multi-document files)
5. Export → DICIE inference
6. Import existing fixtures / eval gold

| Section | What you will see |
|---------|-------------------|
| 0 | Setup |
| 1 | Schemas & taxonomies |
| 2 | Generate synthetic corpus |
| 3 | Seed SQLite store |
| 4 | Inspect claims, docs, fields |
| 5 | Claim bundles |
| 6 | Export for DICIE / training |
| 7 | Run DICIE on corpus export |
| 8 | Import fixtures into the store |
| 9 | Provenance |

Canonical docs: [`docs/sample_document_corpus.md`](../docs/sample_document_corpus.md),
[`docs/data_provenance.md`](../docs/data_provenance.md).

> All documents are **fictional**. Carrier branding (`American Family` / `AmFam`) is used
> only to simulate intake surfaces — no proprietary claim files are loaded.
"""
        ),
        md("## 0. Setup"),
        code(SETUP),
        md("## 1. Schemas & taxonomies"),
        code(
            """
med_schema = read_json(REPO_ROOT / "data" / "schemas" / "medical_bill_skeleton.schema.json")
sal_schema = read_json(REPO_ROOT / "data" / "schemas" / "salvage_document_skeleton.schema.json")
print("Medical schema title:", med_schema["title"])
print("  required:", med_schema["required"])
print("Salvage schema title:", sal_schema["title"])
print("  required:", sal_schema["required"])

for app in ("medical_bills", "salvage_claims"):
    profile = load_application(app)
    print(f"\\n[{app}] labels={profile.labels}")
    print(f"  extraction_fields={profile.extraction_fields}")
"""
        ),
        md("## 2. Generate synthetic corpus"),
        code(
            """
corpus = generate_corpus(
    seed=SEED,
    medical_per_type=6,
    salvage_per_type=6,
    bundles_per_app=2,
    include_canonical_fixtures=True,
)
print(f"claims={len(corpus.claims)}  documents={len(corpus.documents)}")

type_counts = Counter((d.application, d.document_type) for d in corpus.documents)
pd.DataFrame(
    [{"application": a, "document_type": t, "count": n} for (a, t), n in sorted(type_counts.items())]
)
"""
        ),
        md("### Peek at a Letter of Guarantee and an HCFA"),
        code(
            """
log_doc = next(d for d in corpus.documents if d.document_type == "log")
hcfa_doc = next(d for d in corpus.documents if d.document_type == "hcfa")
display(Markdown(f"**{log_doc.document_id}** (`{log_doc.document_type}`)"))
print(log_doc.text[:800])
print("---")
display(Markdown(f"**{hcfa_doc.document_id}** (`{hcfa_doc.document_type}`)"))
print(hcfa_doc.text[:800])
print("\\nLOG ground truth:", log_doc.ground_truth_fields())
print("HCFA ground truth:", hcfa_doc.ground_truth_fields())
"""
        ),
        md("## 3. Seed SQLite store"),
        code(
            """
if DB_PATH.exists():
    DB_PATH.unlink()
store = DocumentStore(DB_PATH)
n = store.bulk_upsert(corpus.documents, claims=corpus.claims)
store.add_provenance(
    stage="notebook_seed",
    source="sample_document_corpus_walkthrough",
    detail={"documents": n, "seed": SEED},
)
summary = store.summary()
display(summary)
pd.DataFrame(summary["by_application_type"])
"""
        ),
        md("## 4. Inspect claims, documents, and fields"),
        code(
            """
docs = store.list_documents(limit=8)
pd.DataFrame(
    [
        {
            "document_id": d.document_id,
            "application": d.application,
            "document_type": d.document_type,
            "claim_id": d.claim_id,
            "split": d.split,
            "source_kind": d.source_kind,
            "n_fields": len(d.fields),
        }
        for d in docs
    ]
)
"""
        ),
        code(
            """
canonical = store.get_document("sal-log-001")
assert canonical is not None
claim = store.get_claim(canonical.claim_id) if canonical.claim_id else None
print("carrier:", claim.carrier_name if claim else None)
print("fields:", canonical.ground_truth_fields())
print(canonical.text)
"""
        ),
        md("## 5. Claim bundles (multi-document salvage / medical files)"),
        code(
            """
import random

rng = random.Random(7)
claim, bundle_docs = generate_claim_bundle(rng, application="salvage_claims", bundle_index=99)
store.upsert_claim(claim)
for d in bundle_docs:
    store.upsert_document(d)

bundled = store.list_documents(claim_id=claim.claim_id)
print(f"claim {claim.claim_id} has {len(bundled)} documents:")
for d in bundled:
    print(f"  - {d.document_id:28s} {d.document_type:8s}  title={d.title}")
"""
        ),
        md("## 6. Export for DICIE / training"),
        code(
            """
salvage_docie = EXPORTS / "salvage_docie.jsonl"
medical_docie = EXPORTS / "medical_docie.jsonl"
clf_all = EXPORTS / "classification_all.jsonl"
ext_all = EXPORTS / "extraction_all.jsonl"

n_sal = store.export_jsonl(salvage_docie, format="docie", application="salvage_claims")
n_med = store.export_jsonl(medical_docie, format="docie", application="medical_bills")
n_clf = store.export_jsonl(clf_all, format="classification")
n_ext = store.export_jsonl(ext_all, format="extraction")
print({"salvage_docie": n_sal, "medical_docie": n_med, "classification": n_clf, "extraction": n_ext})

sample = load_jsonl(salvage_docie)[0]
meta = {k: sample[k] for k in sample if k != "text"}
print(json.dumps(meta, indent=2)[:800])
print("text preview:\\n", sample["text"][:400])
"""
        ),
        md("## 7. Run DICIE on corpus export"),
        code(
            """
out_pred = DEMO / "docie_salvage_predictions.jsonl"
run_file(
    salvage_docie,
    out_pred,
    application="salvage_claims",
    cfg=cfg,
    run_ocr=False,
    limit=20,
)
summary_path = out_pred.with_suffix(".summary.json")
print("summary →", summary_path)
pred_summary = read_json(summary_path)
display(pred_summary)

preds = load_jsonl(out_pred)
pd.DataFrame(
    [
        {
            "record_id": p["record_id"],
            "document_type": p.get("document_type"),
            "needs_human_review": p.get("needs_human_review"),
            "fields": p.get("fields"),
        }
        for p in preds[:12]
    ]
)
"""
        ),
        md("### Single-document DICIE on a generated LOG"),
        code(
            """
pipe = DociePipeline(application="salvage_claims", cfg=cfg, run_ocr=False)
demo_log = store.get_document("sal-log-001")
prediction = pipe.process(record_id=demo_log.document_id, text=demo_log.text)
display(prediction.response_payload())
"""
        ),
        md("## 8. Import existing fixtures / eval gold into the store"),
        code(
            """
fixtures = REPO_ROOT / "tests" / "fixtures" / "sample_docie_documents.jsonl"
eval_set = REPO_ROOT / "data" / "eval" / "docie_eval_set.jsonl"
n_fix = store.import_docie_jsonl(fixtures, source_kind="test_fixture")
n_eval = store.import_docie_jsonl(eval_set, source_kind="docie_eval_gold") if eval_set.exists() else 0
print(f"imported fixtures={n_fix} eval_gold={n_eval}")
store.summary()
"""
        ),
        md("## 9. Provenance"),
        code(
            r'''
with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT event_id, stage, source, detail_json, created_at "
        "FROM provenance_events ORDER BY event_id"
    ).fetchall()
pd.DataFrame([dict(r) for r in rows])
'''
        ),
        md(
            """
## Next notebooks

- [`sample_corpus_sql_integrations.ipynb`](sample_corpus_sql_integrations.ipynb) — every SQL integration surface
- [`sample_corpus_train_test_pipeline.ipynb`](sample_corpus_train_test_pipeline.ipynb) — full train → test pipeline from the store
"""
        ),
    ]
    return write_nb("sample_document_corpus_walkthrough.ipynb", cells)


def build_sql_integrations() -> Path:
    cells = [
        md(
            """
# Sample Corpus SQL Integrations — `DocumentStore`

Deep dive into **every SQL surface** of the medical + salvage sample corpus:

| Section | Focus |
|---------|--------|
| 0 | Setup + open DB |
| 1 | DDL / schema version |
| 2 | Claims table CRUD |
| 3 | Documents upsert + filters |
| 4 | `document_fields` ground truth |
| 5 | Joins: claim ↔ documents ↔ fields |
| 6 | Splits, source_kind, synthetic flags |
| 7 | Provenance events |
| 8 | Raw SQL analytics |
| 9 | Import / export round-trips |
| 10 | Page table hook (optional assets) |

The store uses stdlib `sqlite3` (same pattern as Discord `notes_store.py`) — no ORM required.
"""
        ),
        md("## 0. Setup"),
        code(SETUP),
        code(
            """
# Fresh demo DB for this notebook
if DB_PATH.exists():
    DB_PATH.unlink()
store = DocumentStore(DB_PATH)
corpus = generate_corpus(
    seed=SEED,
    medical_per_type=5,
    salvage_per_type=5,
    bundles_per_app=1,
    include_canonical_fixtures=True,
)
store.bulk_upsert(corpus.documents, claims=corpus.claims)
print(store.summary())
"""
        ),
        md("## 1. DDL / schema version"),
        code(
            """
print("SCHEMA_VERSION =", SCHEMA_VERSION)
print("--- DDL excerpt ---")
print("\\n".join(DDL.strip().splitlines()[:40]))
print("...")

with sqlite3.connect(DB_PATH) as conn:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    ).fetchall()
print("tables:", [t[0] for t in tables])
print("indexes:", [i[0] for i in indexes])
"""
        ),
        md("## 2. Claims table CRUD"),
        code(
            """
claim = ClaimRecord(
    claim_id="CLM-DEMO-SQL-001",
    application="salvage_claims",
    carrier_name="American Family Insurance",
    state="WI",
    date_of_loss="2024-05-10",
    loss_type="collision",
    policy_number="AF-42-1002003",
    insured_name="Jamie Demo",
    metadata={"notebook": "sql_integrations"},
)
store.upsert_claim(claim)
loaded = store.get_claim("CLM-DEMO-SQL-001")
display(loaded.to_dict() if loaded else None)

# Update carrier branding via upsert
claim.carrier_name = "AmFam"
store.upsert_claim(claim)
print("updated carrier:", store.get_claim("CLM-DEMO-SQL-001").carrier_name)
"""
        ),
        md("## 3. Documents upsert + filter APIs"),
        code(
            """
doc = DocumentRecord(
    document_id="sal-demo-log-sql",
    claim_id="CLM-DEMO-SQL-001",
    application="salvage_claims",
    document_type="log",
    title="Demo Letter of Guarantee",
    text=(
        "LETTER OF GUARANTEE\\n"
        "Heartland Bank Title Department\\n"
        "Claim Number: CLM-DEMO-SQL-001\\n"
        "VIN: 1HGCM82633A004352\\n"
        "Year: 2018\\n"
        "Make: Honda\\n"
        "Model: Accord\\n"
        "Payoff Amount: $3,100.00\\n"
    ),
    source_kind="notebook_demo",
    is_synthetic=True,
    split="train",
    skeleton={
        "vehicle": {
            "vin": "1HGCM82633A004352",
            "year": "2018",
            "make": "Honda",
            "model": "Accord",
        }
    },
    fields=[
        FieldRecord("claim_id", "CLM-DEMO-SQL-001"),
        FieldRecord("vin", "1HGCM82633A004352"),
        FieldRecord("year", "2018"),
        FieldRecord("make", "Honda"),
        FieldRecord("model", "Accord"),
    ],
)
store.upsert_document(doc)

filters = [
    {"application": "salvage_claims"},
    {"application": "salvage_claims", "document_type": "log"},
    {"claim_id": "CLM-DEMO-SQL-001"},
    {"split": "train"},
    {"source_kind": "notebook_demo"},
]
for f in filters:
    n = len(store.list_documents(**f))
    print(f"{f} → {n}")
"""
        ),
        md("## 4. `document_fields` ground truth vs extracted roles"),
        code(
            r'''
store.set_fields(
    "sal-demo-log-sql",
    {"payoff_amount": "3100.00", "lienholder": "Heartland Bank"},
    role="annotation",
)
store.set_fields(
    "sal-demo-log-sql",
    {"vin": "1HGCM82633A004352", "make": "Honda"},
    role="extracted",
    confidence=0.91,
)

with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row
    field_rows = conn.execute(
        "SELECT field_name, field_value, field_role, confidence "
        "FROM document_fields WHERE document_id = ? "
        "ORDER BY field_role, field_name",
        ("sal-demo-log-sql",),
    ).fetchall()
pd.DataFrame([dict(r) for r in field_rows])
'''
        ),
        md("## 5. Joins: claim ↔ documents ↔ fields"),
        code(
            r'''
sql = """
SELECT
  c.claim_id,
  c.carrier_name,
  c.loss_type,
  d.document_id,
  d.document_type,
  d.split,
  COUNT(f.field_id) AS n_ground_truth_fields
FROM claims c
JOIN documents d ON d.claim_id = c.claim_id
LEFT JOIN document_fields f
  ON f.document_id = d.document_id AND f.field_role = 'ground_truth'
WHERE c.application = 'salvage_claims'
GROUP BY c.claim_id, d.document_id
ORDER BY c.claim_id, d.document_type
LIMIT 20
"""
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(sql, conn)
df
'''
        ),
        md("## 6. Splits, source kinds, synthetic flags"),
        code(
            r'''
with sqlite3.connect(DB_PATH) as conn:
    split_df = pd.read_sql_query(
        """
        SELECT application, document_type, split, COUNT(*) AS n
        FROM documents
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """,
        conn,
    )
    source_df = pd.read_sql_query(
        """
        SELECT source_kind, is_synthetic, COUNT(*) AS n
        FROM documents
        GROUP BY 1, 2
        """,
        conn,
    )
display(Markdown("### Split distribution"))
display(split_df)
display(Markdown("### Source kinds"))
display(source_df)
'''
        ),
        md("## 7. Provenance events"),
        code(
            r'''
store.add_provenance(
    stage="sql_notebook_demo",
    source="sample_corpus_sql_integrations",
    document_id="sal-demo-log-sql",
    claim_id="CLM-DEMO-SQL-001",
    detail={"action": "annotated_extra_fields"},
)
with sqlite3.connect(DB_PATH) as conn:
    prov = pd.read_sql_query(
        "SELECT event_id, document_id, claim_id, stage, source, detail_json "
        "FROM provenance_events",
        conn,
    )
prov
'''
        ),
        md("## 8. Raw SQL analytics"),
        code(
            r'''
analytics = """
SELECT
  application,
  document_type,
  ROUND(AVG(LENGTH(text)), 1) AS avg_chars,
  MIN(LENGTH(text)) AS min_chars,
  MAX(LENGTH(text)) AS max_chars,
  COUNT(*) AS n
FROM documents
GROUP BY application, document_type
ORDER BY application, document_type
"""
carrier_sql = """
SELECT carrier_name, COUNT(*) AS n_claims
FROM claims
GROUP BY carrier_name
ORDER BY n_claims DESC
"""
missing_vin = """
SELECT d.document_id, d.document_type, f.field_value AS vin
FROM documents d
LEFT JOIN document_fields f
  ON f.document_id = d.document_id
 AND f.field_name = 'vin'
 AND f.field_role = 'ground_truth'
WHERE d.application = 'salvage_claims'
ORDER BY CASE WHEN f.field_value IS NULL OR f.field_value = '' THEN 0 ELSE 1 END,
         d.document_id
LIMIT 15
"""
with sqlite3.connect(DB_PATH) as conn:
    display(Markdown("### Text length by type"))
    display(pd.read_sql_query(analytics, conn))
    display(Markdown("### Carrier mix (synthetic branding)"))
    display(pd.read_sql_query(carrier_sql, conn))
    display(Markdown("### Salvage VIN coverage"))
    display(pd.read_sql_query(missing_vin, conn))
'''
        ),
        md("## 9. Import / export round-trips"),
        code(
            """
roundtrip_path = EXPORTS / "sql_roundtrip_docie.jsonl"
exported = store.export_jsonl(roundtrip_path, format="docie", application="salvage_claims")
print("exported", exported)

# Import into a second DB to prove portability
db2 = DEMO / "documents_roundtrip.db"
if db2.exists():
    db2.unlink()
store2 = DocumentStore(db2)
imported = store2.import_docie_jsonl(roundtrip_path, source_kind="roundtrip")
print("imported", imported)
print(store2.summary())

# Field fidelity check
a = store.get_document("sal-log-001")
b = store2.get_document("sal-log-001")
print("field match:", a.ground_truth_fields() == b.ground_truth_fields() if a and b else None)
"""
        ),
        md("## 10. Optional `document_pages` hook"),
        code(
            r'''
# The pages table is reserved for rendered/OCR page assets (DICIE cache paths, etc.)
now = time.time()
with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        "INSERT INTO document_pages ("
        " document_id, page_index, image_path, width, height, dpi,"
        " ocr_text, words_json, created_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(document_id, page_index) DO UPDATE SET"
        " image_path=excluded.image_path, ocr_text=excluded.ocr_text",
        (
            "sal-demo-log-sql",
            0,
            str(DEMO / "pages" / "sal-demo-log-sql_p0.png"),
            1000,
            1200,
            200,
            "LETTER OF GUARANTEE ...",
            json.dumps([{"text": "LETTER", "bbox": [10, 10, 80, 30]}]),
            now,
        ),
    )
    conn.commit()
    pages = pd.read_sql_query(
        "SELECT document_id, page_index, width, height, dpi, image_path "
        "FROM document_pages",
        conn,
    )
pages
'''
        ),
        md(
            """
## Takeaways

- **Canonical house** for synthetic medical + salvage docs is SQLite via `DocumentStore`
- Ground-truth lives in `document_fields` with roles (`ground_truth` / `extracted` / `annotation`)
- Claim bundles are just multiple `documents` rows sharing `claim_id`
- JSONL import/export keeps DICIE + training pipelines file-compatible
- Raw SQL is available anytime for analytics that the Python API does not wrap
"""
        ),
    ]
    return write_nb("sample_corpus_sql_integrations.ipynb", cells)


def build_train_test_pipeline() -> Path:
    cells = [
        md(
            """
# Sample Corpus Train → Test Pipeline

Full **training and evaluation pipeline** driven by the SQL sample corpus:

```
SQLite DocumentStore
  → prepare classification / extraction splits
  → train TF-IDF + Random Forest (per application)
  → evaluate on held-out test split
  → run DICIE extraction eval on store-exported gold
  → summarize metrics
```

Applications:

- `medical_bills` — `hcfa` / `ub04` / `other`
- `salvage_claims` — `log` / `sales` / `other`

This uses only synthetic data (see [`docs/data_provenance.md`](../docs/data_provenance.md)).
Smoke-scale by default so it finishes quickly on CPU.
"""
        ),
        md("## 0. Setup"),
        code(SETUP),
        code(
            """
SMOKE = True          # Fast RF + small corpus; set False for a larger run
MEDICAL_PER_TYPE = 10 if SMOKE else 30
SALVAGE_PER_TYPE = 10 if SMOKE else 30
BUNDLES = 2 if SMOKE else 4

if DB_PATH.exists():
    DB_PATH.unlink()
store = DocumentStore(DB_PATH)
corpus = generate_corpus(
    seed=SEED,
    medical_per_type=MEDICAL_PER_TYPE,
    salvage_per_type=SALVAGE_PER_TYPE,
    bundles_per_app=BUNDLES,
    include_canonical_fixtures=True,
)
store.bulk_upsert(corpus.documents, claims=corpus.claims)
store.add_provenance(
    stage="train_pipeline_seed",
    source="sample_corpus_train_test_pipeline",
    detail={"smoke": SMOKE, "documents": len(corpus.documents)},
)
display(store.summary())
pd.DataFrame(store.summary()["by_application_type"])
"""
        ),
        md("## 1. Prepare datasets from SQL (classification + extraction)"),
        code(
            """
report = prepare_both_applications(store, PREPARED)
write_json(DEMO / "prepare_report.json", report)
print(json.dumps(report, indent=2)[:2000])

for app in ("medical_bills", "salvage_claims"):
    clf_summary = read_json(PREPARED / app / "classification" / "summary.json")
    ext_summary = read_json(PREPARED / app / "extraction" / "summary.json")
    print(app, "clf counts=", clf_summary["counts"], "ext all=", ext_summary["n_all"])
"""
        ),
        md("### Peek prepared rows + split balance"),
        code(
            """
rows = []
for app in ("medical_bills", "salvage_claims"):
    for split in ("train", "val", "test"):
        path = PREPARED / app / "classification" / f"{split}.jsonl"
        for r in load_jsonl(path):
            rows.append({"application": app, "split": split, "label": r["label"]})
bal = pd.DataFrame(rows)
display(pd.crosstab([bal.application, bal.split], bal.label, margins=True))
"""
        ),
        md("## 2. Train TF-IDF + Random Forest classifiers from store splits"),
        code(
            """
train_metrics = {}
for app in ("medical_bills", "salvage_claims"):
    prepared_dir = PREPARED / app / "classification"
    model_out = MODELS / f"{app}_tfidf_rf.joblib"
    metrics = fit_tfidf_random_forest(
        prepared_dir,
        model_out=model_out,
        random_state=SEED,
        smoke=SMOKE,
    )
    train_metrics[app] = metrics
    print(
        f"{app}: train={metrics['n_train']} val_acc={metrics.get('val_accuracy')} "
        f"test_acc={metrics.get('test_accuracy')} test_macro_f1={metrics.get('test_macro_f1')}"
    )

write_json(DEMO / "rf_train_metrics.json", train_metrics)
pd.DataFrame(
    [
        {
            "application": app,
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "val_accuracy": m.get("val_accuracy"),
            "test_accuracy": m.get("test_accuracy"),
            "test_macro_f1": m.get("test_macro_f1"),
        }
        for app, m in train_metrics.items()
    ]
)
"""
        ),
        md("## 3. Test-set confusion matrices"),
        code(
            """
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, app in zip(axes, ("medical_bills", "salvage_claims")):
    bundle = joblib.load(MODELS / f"{app}_tfidf_rf.joblib")
    pipe = bundle["pipeline"]
    labels = bundle["labels"]
    test_rows = load_jsonl(PREPARED / app / "classification" / "test.jsonl")
    y_true = [r["label"] for r in test_rows]
    y_pred = list(pipe.predict([r["text"] for r in test_rows])) if test_rows else []
    if not test_rows:
        ax.set_title(f"{app} (no test rows)")
        continue
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(app)
plt.tight_layout()
plt.show()
"""
        ),
        md("## 4. DICIE extraction eval on store-exported gold"),
        code(
            """
docie_metrics = {}
for app in ("medical_bills", "salvage_claims"):
    gold = PREPARED / app / "extraction" / "docie_eval.jsonl"
    result = evaluate_application(
        app,
        eval_path=gold,
        cfg=cfg,
        run_ocr=False,
        limit=40 if SMOKE else None,
    )
    docie_metrics[app] = {
        "application": result["application"],
        "n": result["n"],
        "classification_accuracy": result["classification"]["accuracy"],
        "classification_macro_f1": result["classification"]["macro_f1"],
        "extraction_micro_f1": result["extraction"].get("micro_f1"),
        "extraction_macro_f1": result["extraction"].get("macro_f1"),
        "per_field": result["extraction"].get("per_field")
        or result["extraction"].get("fields"),
    }
    print(f"\\n=== {app} ===")
    print(json.dumps(docie_metrics[app], indent=2)[:2000])

write_json(DEMO / "docie_eval_from_store.json", docie_metrics)
pd.DataFrame(
    [
        {
            "application": app,
            "n": m["n"],
            "clf_accuracy": m["classification_accuracy"],
            "clf_macro_f1": m["classification_macro_f1"],
            "ext_micro_f1": m["extraction_micro_f1"],
            "ext_macro_f1": m["extraction_macro_f1"],
        }
        for app, m in docie_metrics.items()
    ]
)
"""
        ),
        md("## 5. Batch DICIE predictions written back as extracted fields"),
        code(
            r'''
# Run salvage predictions and persist extracted fields into SQL (role=extracted)
salvage_export = EXPORTS / "train_pipeline_salvage_docie.jsonl"
store.export_jsonl(
    salvage_export, format="docie", application="salvage_claims", split="test"
)
pred_out = DEMO / "salvage_test_predictions.jsonl"
run_file(
    salvage_export,
    pred_out,
    application="salvage_claims",
    cfg=cfg,
    run_ocr=False,
)

n_updated = 0
for pred in load_jsonl(pred_out):
    rid = pred["record_id"]
    fields = pred.get("fields") or {}
    flat = {}
    for k, v in fields.items():
        if isinstance(v, list):
            flat[k] = v[0] if v else None
        else:
            flat[k] = v
    if not flat:
        continue
    conf = None
    if isinstance(pred.get("extraction"), dict):
        conf = pred["extraction"].get("confidence")
    store.set_fields(
        rid,
        {k: None if v is None else str(v) for k, v in flat.items()},
        role="extracted",
        confidence=conf,
    )
    n_updated += 1
print(f"wrote extracted fields for {n_updated} test documents")

cmp_sql = """
SELECT
  g.document_id,
  g.field_name,
  g.field_value AS ground_truth,
  e.field_value AS extracted
FROM document_fields g
LEFT JOIN document_fields e
  ON e.document_id = g.document_id
 AND e.field_name = g.field_name
 AND e.field_role = 'extracted'
WHERE g.field_role = 'ground_truth'
  AND g.document_id IN (
        SELECT document_id FROM documents
        WHERE split='test' AND application='salvage_claims'
      )
ORDER BY g.document_id, g.field_name
LIMIT 40
"""
with sqlite3.connect(DB_PATH) as conn:
    cmp = pd.read_sql_query(cmp_sql, conn)
cmp
'''
        ),
        md("## 6. Optional DistilBERT smoke train (transformers)"),
        code(
            """
# Optional deep classifier smoke — skipped automatically if deps/data are insufficient.
RUN_TRANSFORMERS_SMOKE = False  # set True on machines with transformers + datasets installed

transformers_metrics = {}
if RUN_TRANSFORMERS_SMOKE:
    try:
        from src.classification.eval import evaluate as eval_clf
        from src.classification.train_classifier import train as train_clf

        for app in ("medical_bills", "salvage_claims"):
            prepared = PREPARED / app / "classification"
            out_dir = MODELS / f"{app}_distilbert_smoke"
            model_dir = train_clf(
                prepared_dir=prepared,
                cfg=cfg,
                output_dir=out_dir,
                smoke=True,
                wandb_settings=None,
            )
            metrics = eval_clf(model_dir=model_dir, prepared_dir=prepared, cfg=cfg)
            transformers_metrics[app] = {
                k: metrics[k]
                for k in metrics
                if k in ("accuracy", "macro_f1", "weighted_f1", "n")
            }
            print(app, transformers_metrics[app])
    except Exception as exc:
        print("Transformers smoke skipped/failed:", exc)
else:
    print("RUN_TRANSFORMERS_SMOKE=False — using TF-IDF RF metrics only")
write_json(DEMO / "transformers_smoke_metrics.json", transformers_metrics)
"""
        ),
        md("## 7. Pipeline scorecard"),
        code(
            """
scorecard = []
for app, m in train_metrics.items():
    scorecard.append(
        {
            "application": app,
            "model": "tfidf_rf",
            "test_accuracy": m.get("test_accuracy"),
            "test_macro_f1": m.get("test_macro_f1"),
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "docie_clf_accuracy": docie_metrics[app]["classification_accuracy"],
            "docie_ext_micro_f1": docie_metrics[app]["extraction_micro_f1"],
        }
    )
score_df = pd.DataFrame(scorecard)
display(score_df)
write_json(
    DEMO / "pipeline_scorecard.json",
    {
        "rf": {
            app: {k: v for k, v in m.items() if k != "val_report" and k != "test_report"}
            for app, m in train_metrics.items()
        },
        "docie": docie_metrics,
        "db_path": str(DB_PATH),
        "prepared": str(PREPARED),
    },
)
print("artifacts under", DEMO)
for p in sorted(DEMO.rglob("*")):
    if p.is_file():
        print(" ", p.relative_to(DEMO))
"""
        ),
        md(
            """
## Recap

1. **Seeded** a synthetic medical + salvage corpus into SQLite  
2. **Prepared** application-specific train/val/test JSONL from SQL splits  
3. **Trained / tested** TF-IDF + Random Forest classifiers per application  
4. **Evaluated** DICIE extraction against store-exported gold  
5. **Wrote** model extractions back into `document_fields` (`role=extracted`)

CLI equivalents:

```bash
python -m src.storage seed --seed 42
python -m src.storage export --format docie --application salvage_claims \\
  --out data/sample_corpus/exports/salvage_docie.jsonl
python -m src.docie.eval --application salvage_claims \\
  --eval-set data/notebook_demo/sample_corpus/prepared/salvage_claims/extraction/docie_eval.jsonl
```
"""
        ),
    ]
    return write_nb("sample_corpus_train_test_pipeline.ipynb", cells)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compile(Path(__file__).read_text(encoding="utf-8"), __file__, "exec")
    build_walkthrough()
    build_sql_integrations()
    build_train_test_pipeline()


if __name__ == "__main__":
    main()
