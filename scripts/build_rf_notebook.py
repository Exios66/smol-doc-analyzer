#!/usr/bin/env python3
"""Generate notebooks/random_forest_text_handwriting_classification.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "random_forest_text_handwriting_classification.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str):
    return nbf.v4.new_code_cell(source.strip("\n"))


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    cells = [
        md(
            """
# Random Forest classification on text & handwriting corpus

Train a **TF-IDF + Random Forest** classifier on the smol-doc-analyzer synthetic insurance document corpus:

- **Typed text** — clean Stage A documents (`data/synthetic/documents/`)
- **Handwriting / OCR surface** — character-garbled noisy variants (`data/synthetic/noisy/`) that approximate OCR'd handwriting and scanned forms

**Targets**

1. Primary: predict `document_type` (ACORD-inspired taxonomy)
2. Secondary: predict surface style (`typed` vs `handwriting_ocr`)

This notebook is a lightweight classical baseline alongside the DeBERTa / ViT deep classifiers.

**Multilayer WandB suite** (section 9) runs the same pipeline as the CLI:

0. corpus profile → 1. capacity sweep → 2. dual heads → 3. surface slices →
4. confidence/ECE → 5. feature importance / confusion pairs

Logged under project `smol-doc-analyzer`, run name `rf-notebook-multilayer`.
Corpus seeding does **not** open a WandB run from this notebook.
"""
        ),
        md("## 1. Setup"),
        code(
            """
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import ConfusionMatrixDisplay

# Allow running the notebook from notebooks/ or repo root
REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "src").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Drop cached src.* modules + stale bytecode. Long-lived kernels can keep an
# older random_forest (pre-DEFAULT_PRESET_NAMES) alive even after disk edits.
_rf_pycache = REPO_ROOT / "src" / "classification" / "__pycache__"
if _rf_pycache.is_dir():
    for _pyc in _rf_pycache.glob("random_forest*.pyc"):
        _pyc.unlink(missing_ok=True)
for _mod_name in list(sys.modules):
    if _mod_name == "src" or _mod_name.startswith("src."):
        del sys.modules[_mod_name]

import src.classification.random_forest as _rf

_rf = importlib.reload(_rf)
DEFAULT_PRESET_NAMES = getattr(
    _rf, "DEFAULT_PRESET_NAMES", tuple(_rf.CAPACITY_PRESETS.keys())
)
SURFACE_HANDWRITING_OCR = _rf.SURFACE_HANDWRITING_OCR
SURFACE_TYPED = _rf.SURFACE_TYPED
as_str_list = _rf.as_str_list
assign_split_column = _rf.assign_split_column
build_document_type_pipeline = _rf.build_document_type_pipeline
ensure_seed_corpus = _rf.ensure_seed_corpus
evaluate_classifier = _rf.evaluate_classifier
load_text_handwriting_corpus = _rf.load_text_handwriting_corpus
save_random_forest_bundle = _rf.save_random_forest_bundle
top_tfidf_feature_importances = _rf.top_tfidf_feature_importances
write_predictions_jsonl = _rf.write_predictions_jsonl

from src.classification.train_random_forest import train as train_rf_multilayer
from src.utils.config import Config
from src.utils.llm_client import DEFAULT_FREE_FALLBACK_MODELS
from src.utils.wandb_utils import load_wandb_settings

# OpenRouter routing for corpus seeding / Stage A–B:
# - On HTTP 402 / "requires more credits", GenerationClient sticky-routes to free models.
# - PREFER_FREE_OPENROUTER=True skips paid GENERATION_MODEL entirely (best when credits are gone).
PREFER_FREE_OPENROUTER = True  # set False to try GENERATION_MODEL first, then free on 402
os.environ.setdefault(
    "OPENROUTER_FREE_FALLBACK_MODELS",
    ",".join(DEFAULT_FREE_FALLBACK_MODELS),
)
if PREFER_FREE_OPENROUTER:
    os.environ["OPENROUTER_PREFER_FREE"] = "1"
else:
    os.environ.setdefault("OPENROUTER_PREFER_FREE", "0")

pd.set_option("display.max_colwidth", 120)
sns.set_theme(style="whitegrid", context="notebook")
cfg = Config.load()
print("repo:", REPO_ROOT)
print("rf module:", _rf.__file__)
print("presets:", DEFAULT_PRESET_NAMES)
print("documents dir:", cfg.document_output_dir)
print("noisy dir:", cfg.noisy_output_dir)
print(f"OpenRouter key set: {bool(cfg.openrouter_api_key)}")
print(f"generation model: {cfg.generation_model}")
print(f"prefer free: {cfg.openrouter_prefer_free}")
print(f"free fallbacks: {list(cfg.openrouter_free_fallback_models)}")
"""
        ),
        md(
            """
## 2. Ensure corpus (text + handwriting/OCR noise)

Uses `SEED_N=2000` typed documents (~4000 rows with OCR variants). Regenerates when the largest existing corpus is smaller than `SEED_N`.

If `OPENROUTER_API_KEY` is set, Stage A/B may call OpenRouter. With the setup cell above, this notebook prefers **free** OpenRouter models (`openrouter/free` and `:free` fallbacks) so generation still works when paid credits are unavailable; otherwise it falls back to templates.
"""
        ),
        code(
            """
# Target typed-document count (noisy OCR variants are generated 1:1 → ~2× rows).
SEED_N = 2000
SEED = 42

# Regenerates when the largest existing corpus is smaller than SEED_N.
corpus_paths = ensure_seed_corpus(n=SEED_N, seed=SEED, log_wandb=False)
display(pd.Series(corpus_paths, name="path").to_frame())
print(f"requested SEED_N={SEED_N:,} | corpus n={corpus_paths.get('n', '?')} | generated={corpus_paths.get('generated')}")
"""
        ),
        md("## 3. Load typed + handwriting/OCR documents"),
        code(
            """
frame = load_text_handwriting_corpus()
frame = assign_split_column(frame)

print(f"rows: {len(frame):,}  |  unique claims: {frame['claim_id'].nunique():,}")
print("sources:", frame.attrs.get("docs_path"), "|", frame.attrs.get("noisy_path"))

display(frame["surface"].value_counts().rename("count").to_frame())
display(
    frame.groupby(["document_type", "surface"])
    .size()
    .unstack(fill_value=0)
    .assign(total=lambda d: d.sum(axis=1))
    .sort_values("total", ascending=False)
)
display(frame["split"].value_counts().rename("count").to_frame())
display(frame.head(3))
"""
        ),
        md("### Sample: typed vs handwriting/OCR text"),
        code(
            """
sample_id = frame.loc[frame["surface"] == SURFACE_TYPED, "record_id"].iloc[0]
pair = frame[frame["record_id"] == sample_id].set_index("surface")
print("record_id:", sample_id)
print("document_type:", pair.iloc[0]["document_type"])
print("\\n--- TYPED ---\\n")
print(pair.loc[SURFACE_TYPED, "text"][:700])
print("\\n--- HANDWRITING / OCR ---\\n")
print(pair.loc[SURFACE_HANDWRITING_OCR, "text"][:700])
"""
        ),
        md(
            """
## 4. Train Random Forest — document type classification

Features: TF-IDF unigrams + bigrams over the document text (typed and OCR surfaces together).
"""
        ),
        code(
            """
train_df = frame[frame["split"] == "train"].reset_index(drop=True)
val_df = frame[frame["split"] == "val"].reset_index(drop=True)
test_df = frame[frame["split"] == "test"].reset_index(drop=True)

# Fit on train + val (classical baseline); hold out test for final prediction
fit_df = pd.concat([train_df, val_df], ignore_index=True)

doc_clf = build_document_type_pipeline(
    n_estimators=300,
    max_features=20000,
    ngram_range=(1, 2),
    random_state=42,
)
doc_clf.fit(fit_df["text"], fit_df["document_type"])

print(f"fitted on {len(fit_df):,} rows  |  classes: {list(doc_clf.classes_)}")
print(f"TF-IDF vocabulary size: {len(doc_clf.named_steps['tfidf'].vocabulary_):,}")
"""
        ),
        md("## 5. Evaluate on held-out test set"),
        code(
            """
doc_metrics = evaluate_classifier(
    doc_clf,
    as_str_list(test_df["text"]),
    as_str_list(test_df["document_type"]),
    labels=list(doc_clf.classes_),
)

print(f"Test accuracy : {doc_metrics['accuracy']:.4f}")
print(f"Macro F1      : {doc_metrics['macro_f1']:.4f}")
print(f"Weighted F1   : {doc_metrics['weighted_f1']:.4f}")
print(f"N test rows   : {doc_metrics['n']}")

report_df = pd.DataFrame(doc_metrics["classification_report"]).T
display(report_df.round(3))
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(
    confusion_matrix=np.asarray(doc_metrics["confusion_matrix"]),
    display_labels=doc_metrics["labels"],
).plot(ax=ax, colorbar=True, cmap="Blues")
# sklearn stubs type xticks_rotation as str only; set degrees on the axes instead.
ax.tick_params(axis="x", labelrotation=45)
ax.set_title("Document-type Random Forest — test confusion matrix")
plt.tight_layout()
plt.show()
"""
        ),
        md("### Accuracy by surface (typed vs handwriting/OCR)"),
        code(
            """
surface_rows = []
for surface, group in test_df.groupby("surface"):
    m = evaluate_classifier(
        doc_clf,
        as_str_list(group["text"]),
        as_str_list(group["document_type"]),
        labels=list(doc_clf.classes_),
    )
    surface_rows.append(
        {
            "surface": surface,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
        }
    )
surface_metrics = pd.DataFrame(surface_rows).sort_values("surface")
display(surface_metrics.round(4))

fig, ax = plt.subplots(figsize=(6, 3.5))
sns.barplot(data=surface_metrics, x="surface", y="accuracy", hue="surface", ax=ax, legend=False)
ax.set_ylim(0, 1.05)
ax.set_title("Document-type accuracy by text surface")
ax.set_ylabel("accuracy")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 6. Predictions

Generate class predictions + confidence for every test document.
"""
        ),
        code(
            """
y_pred = doc_metrics["predictions"]
y_conf = doc_metrics["max_proba"]

# Build columns explicitly to avoid brittle pandas-stubs rename overloads.
pred_df = pd.DataFrame(
    {
        "record_id": test_df["record_id"].to_numpy(),
        "claim_id": test_df["claim_id"].to_numpy(),
        "surface": test_df["surface"].to_numpy(),
        "true_label": test_df["document_type"].to_numpy(),
        "predicted_label": y_pred,
        "confidence": y_conf,
    }
)
pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]

print("prediction accuracy:", round(float(pred_df["correct"].to_numpy(dtype=float).mean()), 4))
display(pred_df.head(10))

mistakes = pred_df.loc[~pred_df["correct"]].sort_values(by="confidence", ascending=False)
print(f"\\nmisclassified: {len(mistakes)} / {len(pred_df)}")
display(mistakes.head(10))
"""
        ),
        md("## 7. Top TF-IDF features (Random Forest importance)"),
        code(
            """
fi = top_tfidf_feature_importances(doc_clf, top_k=25)
display(fi)

fig, ax = plt.subplots(figsize=(8, 7))
sns.barplot(data=fi, y="feature", x="importance", color="#2a6f97", ax=ax)
ax.set_title("Top TF-IDF features by Random Forest importance")
ax.set_xlabel("mean decrease in impurity")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 8. Secondary model — surface style (typed vs handwriting/OCR)

A second Random Forest predicts whether the text looks typed/clean or OCR/handwriting-noisy.
"""
        ),
        code(
            """
surface_clf = build_document_type_pipeline(
    n_estimators=200,
    max_features=15000,
    ngram_range=(1, 2),
    random_state=42,
)
surface_clf.fit(as_str_list(fit_df["text"]), as_str_list(fit_df["surface"]))

surface_eval = evaluate_classifier(
    surface_clf,
    as_str_list(test_df["text"]),
    as_str_list(test_df["surface"]),
    labels=[SURFACE_TYPED, SURFACE_HANDWRITING_OCR],
)
print(f"Surface accuracy : {surface_eval['accuracy']:.4f}")
print(f"Surface macro F1 : {surface_eval['macro_f1']:.4f}")
display(pd.DataFrame(surface_eval["classification_report"]).T.round(3))

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay(
    confusion_matrix=np.asarray(surface_eval["confusion_matrix"]),
    display_labels=surface_eval["labels"],
).plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title("Surface classifier (typed vs handwriting/OCR)")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 9. Multilayer train + WandB (official experiment)

Runs the same multilayer suite as `python -m src.classification.train_random_forest`:

capacity sweep (`shallow` / `balanced` / `deep` / `char_robust`) → best model →
surface head → typed/OCR slices → confidence/ECE → feature importance.

Artifacts land in `models/random_forest_classifier/` and metrics go to WandB under
namespaced keys (`data/`, `sweep/`, `best/`, `slice/`, `confidence/`, `interp/`).
"""
        ),
        code(
            '''
wb_settings = load_wandb_settings()
out_dir = train_rf_multilayer(
    ensure_data=True,
    seed_n=SEED_N,
    presets=list(DEFAULT_PRESET_NAMES),
    wandb_settings=wb_settings,
    wandb_run_name="rf-notebook-multilayer",
)
print("multilayer artifacts ->", out_dir)
print(
    "WandB:",
    f"enabled={wb_settings.enabled}",
    f"mode={wb_settings.mode}",
    f"project={wb_settings.project}",
)
for name in (
    "sweep_results.json",
    "layer_diagnostics.json",
    "eval_metrics.json",
    "train_meta.json",
):
    path = out_dir / name
    print(f"  {name}: {'ok' if path.exists() else 'MISSING'}")
'''
        ),
        md(
            """
## 10. Quick inference helper

Predict document type for an arbitrary snippet (typed or OCR-like text).
"""
        ),
        code(
            '''
def predict_document(text: str, model=doc_clf, top_k: int = 5) -> dict:
    """Return predicted label, confidence, and top-k class probabilities."""
    proba = model.predict_proba([text])[0]
    ranking = sorted(
        zip(model.classes_, map(float, proba)),
        key=lambda x: x[1],
        reverse=True,
    )
    label, conf = ranking[0]
    return {
        "predicted_label": str(label),
        "confidence": float(conf),
        "top_k": [{"label": str(c), "probability": float(p)} for c, p in ranking[:top_k]],
        # keep top3 for backward compatibility with earlier notebook cells / callers
        "top3": [(str(c), float(p)) for c, p in ranking[:3]],
    }


def display_prediction(result: dict, *, title: str = "Document-type prediction") -> None:
    """Pretty-print a predict_document() result in the notebook."""
    from IPython.display import Markdown, display

    label = result["predicted_label"]
    conf = float(result["confidence"])
    rows = result.get("top_k") or [
        {"label": c, "probability": p} for c, p in result.get("top3", [])
    ]
    top_df = pd.DataFrame(rows)
    top_df["probability_pct"] = (top_df["probability"] * 100).round(1)
    top_df["bar"] = top_df["probability"].map(
        lambda p: "█" * int(round(p * 20)) + "░" * (20 - int(round(p * 20)))
    )

    display(
        Markdown(
            f"### {title}\\n\\n"
            f"**Predicted:** `{label}`  \\n"
            f"**Confidence:** {conf:.1%} ({conf:.4f})"
        )
    )
    display(
        top_df.rename(
            columns={
                "label": "class",
                "probability": "p",
                "probability_pct": "p (%)",
                "bar": "distribution",
            }
        )[["class", "p", "p (%)", "distribution"]]
        .style.format({"p": "{:.4f}", "p (%)": "{:.1f}"})
        .hide(axis="index")
    )

    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.45 * len(top_df))))
    ax.barh(
        top_df["label"][::-1],
        top_df["probability"][::-1],
        color="#3b82f6",
        edgecolor="none",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title(f"Top-{len(top_df)} class probabilities")
    for y, p in enumerate(top_df["probability"][::-1]):
        ax.text(min(p + 0.02, 0.98), y, f"{p:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


demo = """PROPERTY LOSS NOTICE
Claim Number: CLM-2026-111222
Date of Loss: 2025-11-03
Loss Type: water
Description of Loss: Pipe burst in upstairs bathroom; ceiling damage reported.
Reported By: Jordan Lee
"""

result = predict_document(demo)
display_prediction(result)
'''
        ),
    ]

    nb.cells = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
