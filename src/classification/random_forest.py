"""TF-IDF + Random Forest document-type classification for the synthetic corpus.

Supports clean typed text and OCR / handwriting-style noisy variants produced by
``src.generation.noise_injection``. Designed for notebook and CLI reuse.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils.config import Config
from src.utils.io import load_jsonl, read_json, write_json


SURFACE_TYPED = "typed"
SURFACE_HANDWRITING_OCR = "handwriting_ocr"

_CORPUS_N_RE = re.compile(r"_n(\d+)(?:_|\.|$)")


def corpus_n_from_path(path: Path) -> int:
    """Parse ``n`` from filenames like ``documents_from_skeletons_n2000_seed42.jsonl``."""
    match = _CORPUS_N_RE.search(path.stem)
    return int(match.group(1)) if match else 0


def _largest_corpus_path(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda p: (corpus_n_from_path(p), p.stat().st_mtime))


def _rows_from_docs(docs: list[dict[str, Any]], surface: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in docs:
        text = (doc.get("text") or "").strip()
        label = doc.get("document_type")
        if not text or not label:
            continue
        rows.append(
            {
                "record_id": doc.get("record_id"),
                "claim_id": doc.get("claim_id"),
                "text": text,
                "document_type": label,
                "surface": surface,
                "is_noisy": bool(doc.get("is_noisy", surface == SURFACE_HANDWRITING_OCR)),
            }
        )
    return rows


def load_text_handwriting_corpus(
    docs_path: Path | None = None,
    noisy_path: Path | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Load clean typed documents plus OCR/handwriting-noisy variants."""
    cfg = cfg or Config.load()
    if docs_path is None:
        candidates = list(cfg.document_output_dir.glob("documents_from_skeletons_*.jsonl"))
        docs_path = _largest_corpus_path(candidates)
        if docs_path is None:
            raise FileNotFoundError(
                f"No document JSONL under {cfg.document_output_dir}. "
                "Run: python -m src.generation.run_seed_pipeline --n 2000 --seed 42"
            )
    if noisy_path is None:
        stem = docs_path.stem
        noisy_path = cfg.noisy_output_dir / f"noisy_from_{stem}.jsonl"
        if not noisy_path.exists():
            noisy_candidates = list(cfg.noisy_output_dir.glob("noisy_from_*.jsonl"))
            noisy_path = _largest_corpus_path(noisy_candidates)
            if noisy_path is None:
                raise FileNotFoundError(
                    f"No noisy JSONL under {cfg.noisy_output_dir}. "
                    "Run the seed pipeline (includes noise injection)."
                )

    clean = _rows_from_docs(load_jsonl(docs_path), SURFACE_TYPED)
    noisy = _rows_from_docs(load_jsonl(noisy_path), SURFACE_HANDWRITING_OCR)
    frame = pd.DataFrame(clean + noisy)
    frame.attrs["docs_path"] = str(docs_path)
    frame.attrs["noisy_path"] = str(noisy_path)
    return frame


def _split_map_for_record_ids(
    record_ids: list[str],
    labels_by_id: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign train/val/test once per unique ``record_id`` (no surface leakage)."""
    unique_ids = list(dict.fromkeys(record_ids))
    if not unique_ids:
        return {}
    if len(unique_ids) == 1:
        return {unique_ids[0]: "train"}

    id_frame = pd.DataFrame({"record_id": unique_ids})
    if labels_by_id:
        id_frame["document_type"] = id_frame["record_id"].map(labels_by_id)
        counts = id_frame["document_type"].value_counts(dropna=True)
        can_stratify = bool(
            len(id_frame) >= 10
            and not id_frame["document_type"].isna().any()
            and len(counts) > 0
            and counts.min() >= 2
        )
        stratify = id_frame["document_type"] if can_stratify else None
    else:
        stratify = None

    train_ids, temp_ids = train_test_split(
        id_frame["record_id"],
        test_size=0.3,
        random_state=42,
        stratify=stratify,
    )
    temp_list = list(temp_ids)
    if len(temp_list) == 1:
        val_ids, test_ids = temp_list, []
    else:
        temp_labels = (
            [labels_by_id.get(rid) for rid in temp_list] if labels_by_id else None
        )
        can_stratify_temp = False
        if temp_labels is not None and None not in temp_labels:
            temp_counts = pd.Series(temp_labels).value_counts()
            can_stratify_temp = bool(len(temp_list) >= 4 and temp_counts.min() >= 2)
        val_ids, test_ids = train_test_split(
            temp_list,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels if can_stratify_temp else None,
        )

    mapping: dict[str, str] = {rid: "train" for rid in train_ids}
    mapping.update({rid: "val" for rid in val_ids})
    mapping.update({rid: "test" for rid in test_ids})
    return mapping


def assign_split_column(frame: pd.DataFrame, splits_path: Path | None = None) -> pd.DataFrame:
    """Attach train/val/test using project splits when available; else stratified holdout.

    Splits are assigned per unique ``record_id`` so typed and handwriting/OCR
    surfaces of the same document always share a split (no train/test leakage).
    """
    cfg = Config.load()
    path = splits_path or cfg.splits_path
    out = frame.copy()
    # Object dtype avoids TypeError when assigning string labels over all-NaN float columns.
    out["split"] = pd.Series([pd.NA] * len(out), dtype="object", index=out.index)

    labels_by_id: dict[str, str] = {}
    if "document_type" in out.columns:
        for rid, label in zip(out["record_id"].astype(str), out["document_type"].astype(str)):
            labels_by_id.setdefault(rid, label)

    id_to_split: dict[str, str] = {}
    if path.exists():
        splits = read_json(path)
        for split_name in ("train", "val", "test"):
            for rid in splits.get(split_name, []):
                id_to_split[str(rid)] = split_name

    mapped = out["record_id"].astype(str).map(id_to_split)
    out["split"] = mapped.astype("object")
    missing_mask = out["split"].isna()

    # Fallback when splits.json is missing, matches nothing, or only covers some IDs.
    if missing_mask.any():
        missing_ids = out.loc[missing_mask, "record_id"].astype(str).tolist()
        fallback = _split_map_for_record_ids(missing_ids, labels_by_id=labels_by_id)
        out.loc[missing_mask, "split"] = (
            out.loc[missing_mask, "record_id"].astype(str).map(fallback)
        )

    # Final safety: any still-missing IDs go to train (should be rare).
    still_missing = out["split"].isna()
    if still_missing.any():
        out.loc[still_missing, "split"] = "train"

    out["split"] = out["split"].astype(str)
    return out


# Named capacity presets for the multilayer sweep (Layer 1).
CAPACITY_PRESETS: dict[str, dict[str, Any]] = {
    "shallow": {
        "n_estimators": 100,
        "max_depth": 12,
        "max_features": 10000,
        "ngram_range": (1, 2),
        "analyzer": "word",
        "min_df": 2,
        "min_samples_leaf": 2,
        "description": "Shallow trees; fast baseline",
    },
    "balanced": {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": 20000,
        "ngram_range": (1, 2),
        "analyzer": "word",
        "min_df": 2,
        "min_samples_leaf": 1,
        "description": "Default capacity (unlimited depth)",
    },
    "deep": {
        "n_estimators": 500,
        "max_depth": None,
        "max_features": 40000,
        "ngram_range": (1, 2),
        "analyzer": "word",
        "min_df": 2,
        "min_samples_leaf": 1,
        "description": "More trees + larger vocab",
    },
    "char_robust": {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": 30000,
        "ngram_range": (3, 5),
        "analyzer": "char_wb",
        "min_df": 2,
        "min_samples_leaf": 1,
        "description": "Char n-grams for OCR/handwriting robustness",
    },
    "hybrid_ocr": {
        "n_estimators": 400,
        "max_depth": None,
        "max_features": 25000,
        "ngram_range": (1, 2),
        "analyzer": "hybrid",
        "min_df": 2,
        "min_samples_leaf": 1,
        "description": "Word + char_wb FeatureUnion for typed+OCR generalization",
    },
}

DEFAULT_PRESET_NAMES: tuple[str, ...] = (
    "shallow",
    "balanced",
    "char_robust",
    "hybrid_ocr",
    "deep",
)

# Tiny presets for CI / --smoke (same keys, fewer trees).
SMOKE_CAPACITY_PRESETS: dict[str, dict[str, Any]] = {
    name: {
        **{k: v for k, v in cfg.items() if k != "n_estimators"},
        "n_estimators": 60,
        "max_features": min(int(cfg.get("max_features") or 2000), 2000),
        "min_df": 1,
        "description": f"Smoke: {cfg.get('description', name)}",
    }
    for name, cfg in CAPACITY_PRESETS.items()
}


def _make_vectorizer(
    *,
    analyzer: str,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
) -> TfidfVectorizer | FeatureUnion:
    """Build TF-IDF (word / char) or a hybrid FeatureUnion for OCR-robust RF."""
    common = dict(
        lowercase=True,
        strip_accents="unicode",
        min_df=min_df,
        sublinear_tf=True,
    )
    if analyzer == "hybrid":
        word_feats = max(1000, max_features // 2)
        char_feats = max(1000, max_features // 2)
        return FeatureUnion(
            transformer_list=[
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=ngram_range,
                        max_features=word_feats,
                        **common,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=char_feats,
                        **common,
                    ),
                ),
            ]
        )
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer=analyzer,
        **common,
    )


def build_document_type_pipeline(
    n_estimators: int = 200,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    random_state: int = 42,
    *,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    min_df: int = 2,
    analyzer: str = "word",
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                _make_vectorizer(
                    analyzer=analyzer,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                ),
            ),
        ]
    )


def set_vectorizer_min_df(model: Pipeline, min_df: int = 1) -> None:
    """Set ``min_df`` on a TF-IDF step or hybrid FeatureUnion."""
    vec = model.named_steps["tfidf"]
    if isinstance(vec, FeatureUnion):
        vec.set_params(word__min_df=min_df, char__min_df=min_df)
    else:
        vec.set_params(min_df=min_df)


def pipeline_from_preset(
    preset_name: str,
    *,
    smoke: bool = False,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Build a pipeline from a named capacity preset. Returns (pipeline, preset_config)."""
    catalog = SMOKE_CAPACITY_PRESETS if smoke else CAPACITY_PRESETS
    if preset_name not in catalog:
        known = ", ".join(sorted(catalog))
        raise ValueError(f"Unknown preset {preset_name!r}. Known: {known}")
    cfg = dict(catalog[preset_name])
    pipe = build_document_type_pipeline(
        n_estimators=int(cfg["n_estimators"]),
        max_features=int(cfg["max_features"]),
        ngram_range=tuple(cfg["ngram_range"]),  # type: ignore[arg-type]
        random_state=random_state,
        max_depth=cfg.get("max_depth"),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
        min_df=int(cfg.get("min_df", 2)),
        analyzer=str(cfg.get("analyzer", "word")),
    )
    return pipe, {**cfg, "preset": preset_name}


def fit_pipeline_with_tree_curve(
    model: Pipeline,
    texts: list[str] | pd.Series,
    y: list[str] | pd.Series,
    *,
    val_texts: list[str] | pd.Series | None = None,
    val_y: list[str] | pd.Series | None = None,
    tree_chunk: int = 25,
) -> list[dict[str, Any]]:
    """Fit TF-IDF once, then grow RF trees in chunks (OOB + optional val curve).

    Returns a history list suitable for stepped WandB logging:
    ``n_estimators``, ``oob_score``, optional ``val_accuracy`` / ``val_macro_f1``.
    """
    from sklearn.utils.class_weight import compute_class_weight

    vectorizer = model.named_steps["tfidf"]
    forest: RandomForestClassifier = model.named_steps["rf"]
    y_list = list(y)
    x_train = vectorizer.fit_transform(texts)
    x_val = vectorizer.transform(val_texts) if val_texts is not None else None

    target_trees = int(forest.n_estimators)
    chunk = max(1, int(tree_chunk))
    n_rows = int(x_train.shape[0])
    # OOB is noisy with tiny forests; only enable once we have a reasonable tree count.
    oob_min_trees = 30
    classes = np.unique(y_list)
    weights = compute_class_weight("balanced", classes=classes, y=y_list)
    class_weight = {cls: float(w) for cls, w in zip(classes, weights)}

    forest.set_params(
        warm_start=True,
        oob_score=False,
        n_estimators=0,
        class_weight=class_weight,
    )

    history: list[dict[str, Any]] = []
    n_trees = 0
    while n_trees < target_trees:
        n_trees = min(target_trees, n_trees + chunk)
        enable_oob = n_rows >= 10 and n_trees >= oob_min_trees
        forest.set_params(n_estimators=n_trees, oob_score=enable_oob)
        forest.fit(x_train, y_list)
        oob = None
        if enable_oob and hasattr(forest, "oob_score_"):
            oob = float(forest.oob_score_)
        row: dict[str, Any] = {
            "n_estimators": n_trees,
            "oob_score": oob,
        }
        if x_val is not None and val_y is not None and len(list(val_y)) > 0:
            pred = forest.predict(x_val)
            row["val_accuracy"] = float(accuracy_score(val_y, pred))
            row["val_macro_f1"] = float(
                f1_score(val_y, pred, average="macro", zero_division=0)
            )
        history.append(row)

    forest.set_params(warm_start=False)
    return history


def as_str_list(values: Any) -> list[str]:
    """Normalize pandas column selections / array-likes to ``list[str]``.

    Use at call sites when basedpyright types ``df[col]`` as
    ``Series | DataFrame | ndarray | Any`` (too wide for narrow annotations).
    """
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise TypeError(
                f"Expected a single column for labels/texts, got shape {values.shape}"
            )
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        return [str(v) for v in values.fillna("").tolist()]
    return [str(v) for v in list(values)]


def evaluate_classifier(
    model: Pipeline,
    texts: Any,
    y_true: Any,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    # Normalize at the boundary so callers may pass Series / list / ndarray.
    text_values = as_str_list(texts)
    true_values = as_str_list(y_true)
    y_pred = model.predict(text_values)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(text_values)
    label_list = labels or sorted(set(true_values) | set(list(y_pred)))
    report = classification_report(
        true_values, y_pred, labels=label_list, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true_values, y_pred, labels=label_list)
    metrics = {
        "accuracy": float(accuracy_score(true_values, y_pred)),
        "macro_f1": float(f1_score(true_values, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(true_values, y_pred, average="weighted", zero_division=0)
        ),
        "n": int(len(true_values)),
        "labels": label_list,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": list(y_pred),
    }
    if y_proba is not None:
        metrics["max_proba"] = y_proba.max(axis=1).tolist()
        metrics["classes"] = list(model.classes_)
    return metrics


def top_tfidf_feature_importances(model: Pipeline, top_k: int = 30) -> pd.DataFrame:
    """Map Random Forest importances back to TF-IDF feature names."""
    vectorizer = model.named_steps["tfidf"]
    forest: RandomForestClassifier = model.named_steps["rf"]
    names = np.asarray(vectorizer.get_feature_names_out())
    importances = forest.feature_importances_
    order = np.argsort(importances)[::-1][:top_k]
    return pd.DataFrame(
        {
            "feature": names[order],
            "importance": importances[order],
        }
    )


def profile_corpus(frame: pd.DataFrame) -> dict[str, Any]:
    """Layer 0: corpus profile for ``data/*`` WandB metrics."""
    n = int(len(frame))
    surface_counts = frame["surface"].value_counts().to_dict() if n else {}
    type_counts = frame["document_type"].value_counts().to_dict() if n else {}
    split_counts = (
        frame["split"].value_counts().to_dict() if n and "split" in frame.columns else {}
    )
    typed_n = int(surface_counts.get(SURFACE_TYPED, 0))
    ocr_n = int(surface_counts.get(SURFACE_HANDWRITING_OCR, 0))
    return {
        "n_rows": n,
        "n_classes": int(frame["document_type"].nunique()) if n else 0,
        "n_typed": typed_n,
        "n_ocr": ocr_n,
        "typed_frac": float(typed_n / n) if n else 0.0,
        "ocr_frac": float(ocr_n / n) if n else 0.0,
        "surface_counts": {str(k): int(v) for k, v in surface_counts.items()},
        "class_counts": {str(k): int(v) for k, v in type_counts.items()},
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
    }


def evaluate_by_surface(
    model: Pipeline,
    frame: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "document_type",
    surface_col: str = "surface",
) -> dict[str, dict[str, Any]]:
    """Layer 3: document-type metrics sliced by surface (typed vs OCR)."""
    out: dict[str, dict[str, Any]] = {}
    for surface, group in frame.groupby(surface_col):
        if group.empty:
            continue
        metrics = evaluate_classifier(
            model,
            group[text_col],
            group[label_col],
            labels=list(model.classes_),
        )
        # Drop bulky fields from slice summary; keep headline + n
        out[str(surface)] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "n": metrics["n"],
            "labels": metrics["labels"],
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
    return out


def confidence_diagnostics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Layer 4: confidence bins + expected calibration error (ECE-lite)."""
    if not confidences:
        return {
            "mean_confidence": 0.0,
            "mean_confidence_correct": 0.0,
            "mean_confidence_incorrect": 0.0,
            "ece": 0.0,
            "bins": [],
        }
    conf = np.asarray(list(confidences), dtype=float)
    correct = np.asarray([t == p for t, p in zip(y_true, y_pred)], dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Include 1.0 in the last bin
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "n": 0,
                    "accuracy": 0.0,
                    "mean_confidence": 0.0,
                }
            )
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "n": count,
                "accuracy": bin_acc,
                "mean_confidence": bin_conf,
            }
        )
    return {
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else 0.0,
        "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else 0.0,
        "ece": float(ece),
        "bins": bins,
    }


def top_confusion_pairs(
    metrics: dict[str, Any],
    *,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Off-diagonal confusion pairs sorted by count (interpretability)."""
    labels = list(metrics.get("labels") or [])
    cm = np.asarray(metrics.get("confusion_matrix") or [], dtype=int)
    if cm.size == 0 or not labels or cm.shape[0] != len(labels):
        return []
    pairs: list[dict[str, Any]] = []
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append({"true": true_lab, "pred": pred_lab, "count": count})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_k]


def ensure_seed_corpus(
    n: int = 2000,
    seed: int = 42,
    *,
    log_wandb: bool = False,
) -> dict[str, str]:
    """Generate the synthetic corpus if missing or smaller than ``n``.

    Reuses an existing JSONL only when its parsed sample size is ``>= n``.
    Nested generation does **not** open a WandB run by default so notebook / RF
    training runs are not mis-attributed as ``seed_pipeline`` experiments.
    Pass ``log_wandb=True`` (or call ``run_seed_pipeline`` directly) to track
    corpus generation separately.
    """
    cfg = Config.load()
    existing = list(cfg.document_output_dir.glob("documents_from_skeletons_*.jsonl"))
    best = _largest_corpus_path(existing)
    if best is not None and corpus_n_from_path(best) >= n:
        noisy_candidates = list(cfg.noisy_output_dir.glob("noisy_from_*.jsonl"))
        noisy = cfg.noisy_output_dir / f"noisy_from_{best.stem}.jsonl"
        if not noisy.exists():
            noisy_best = _largest_corpus_path(noisy_candidates)
            noisy_path = str(noisy_best) if noisy_best else ""
        else:
            noisy_path = str(noisy)
        return {
            "documents": str(best),
            "noisy": noisy_path,
            "generated": "false",
            "n": str(corpus_n_from_path(best)),
        }
    from src.generation.run_seed_pipeline import run_seed
    from src.utils.wandb_utils import load_wandb_settings

    settings = load_wandb_settings(enabled=log_wandb)
    paths = run_seed(n=n, seed=seed, skip_ingest=True, wandb_settings=settings)
    paths["generated"] = "true"
    paths["n"] = str(n)
    return paths


def _label_ids(labels: Sequence[str], values: Sequence[str]) -> list[int] | None:
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    ids: list[int] = []
    for value in values:
        idx = label_to_id.get(value)
        if idx is None:
            return None
        ids.append(idx)
    return ids


def _per_class_rows(metrics: dict[str, Any]) -> list[list[Any]]:
    labels = list(metrics.get("labels") or [])
    report = metrics.get("classification_report") or {}
    rows: list[list[Any]] = []
    for label in labels:
        stats = report.get(label) or {}
        if isinstance(stats, dict):
            rows.append(
                [
                    label,
                    float(stats.get("precision", 0.0)),
                    float(stats.get("recall", 0.0)),
                    float(stats.get("f1-score", 0.0)),
                    int(stats.get("support", 0)),
                ]
            )
    return rows


def log_random_forest_to_wandb(
    *,
    doc_metrics: dict[str, Any],
    surface_metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    y_true: Sequence[str] | None = None,
    artifact_paths: Sequence[Path] | None = None,
    wandb_settings: Any | None = None,
    run_name: str | None = None,
    layer_payload: dict[str, Any] | None = None,
) -> None:
    """Log RF multilayer metrics to Weights & Biases as many stepped points.

    WandB charts use ``_step`` as the x-axis. Logging everything in one call
    yields a single point; this function emits one step per layer datapoint
    (corpus → sweep → tree curve → per-class → confidence bins → features).
    """
    from src.utils.wandb_utils import start_run

    payload = layer_payload or {}
    labels = list(doc_metrics.get("labels") or [])
    preds = list(doc_metrics.get("predictions") or [])
    name = run_name or "rf-multilayer-random-forest-classifier"
    run_config = {
        "task": "document_type_random_forest_multilayer",
        "model": "sklearn.RandomForestClassifier",
        **(config or {}),
    }
    tags = ["classification", "random_forest", "multilayer"]
    with start_run(
        name=name,
        job_type="train",
        config=run_config,
        tags=tags,
        settings=wandb_settings,
    ) as wb:
        step = 0
        data_profile = payload.get("data_profile") or {}
        slice_metrics = payload.get("slice_metrics") or {}
        typed_slice = slice_metrics.get(SURFACE_TYPED) or {}
        ocr_slice = slice_metrics.get(SURFACE_HANDWRITING_OCR) or {}
        conf = payload.get("confidence") or {}

        # ---- Layer 0: corpus scalars (one step) ----
        data_log: dict[str, Any] = {"layer": 0}
        for key in ("n_rows", "n_classes", "n_typed", "n_ocr", "typed_frac", "ocr_frac"):
            if key in data_profile and isinstance(data_profile[key], (int, float)):
                data_log[f"data/{key}"] = data_profile[key]
        wb.log(data_log, step=step)
        step += 1

        # Class counts as a stepped series (one class per step)
        class_counts = data_profile.get("class_counts") or {}
        for class_name, count in sorted(class_counts.items()):
            wb.log(
                {
                    "data/class_count": int(count),
                    "data/class_rank": step,
                },
                step=step,
            )
            step += 1
        if class_counts:
            wb.log_table(
                "data/class_counts",
                ["document_type", "count"],
                [[k, v] for k, v in sorted(class_counts.items())],
            )

        # ---- Layer 1: capacity sweep (one step per preset) ----
        sweep_rows = payload.get("sweep_rows") or []
        for i, row in enumerate(sweep_rows):
            point: dict[str, Any] = {
                "layer": 1,
                "sweep/preset_idx": i,
                "sweep/n_estimators": row.get("n_estimators"),
            }
            for metric_key in ("val_accuracy", "val_macro_f1", "val_weighted_f1"):
                val = row.get(metric_key)
                if isinstance(val, (int, float)):
                    # Shared keys → overlayable charts across presets
                    point[f"sweep/{metric_key}"] = val
                    preset = row.get("preset")
                    if preset:
                        point[f"sweep/{preset}/{metric_key}"] = val
            wb.log({k: v for k, v in point.items() if v is not None}, step=step)
            step += 1
        if sweep_rows:
            wb.log_table(
                "sweep/results",
                [
                    "preset",
                    "val_accuracy",
                    "val_macro_f1",
                    "val_weighted_f1",
                    "n_estimators",
                    "analyzer",
                    "max_depth",
                ],
                [
                    [
                        r.get("preset"),
                        r.get("val_accuracy"),
                        r.get("val_macro_f1"),
                        r.get("val_weighted_f1"),
                        r.get("n_estimators"),
                        r.get("analyzer"),
                        r.get("max_depth"),
                    ]
                    for r in sweep_rows
                ],
            )

        # ---- Layer 2a: tree-growth learning curve ----
        tree_history = payload.get("tree_history") or []
        for row in tree_history:
            point = {
                "layer": 2,
                "train/n_estimators": row.get("n_estimators"),
            }
            if isinstance(row.get("oob_score"), (int, float)):
                point["train/oob_score"] = row["oob_score"]
            if isinstance(row.get("val_accuracy"), (int, float)):
                point["train/val_accuracy"] = row["val_accuracy"]
            if isinstance(row.get("val_macro_f1"), (int, float)):
                point["train/val_macro_f1"] = row["val_macro_f1"]
            wb.log(point, step=step)
            step += 1

        # ---- Layer 2b: best + surface headlines ----
        best_log: dict[str, Any] = {
            "layer": 2,
            "test/accuracy": doc_metrics["accuracy"],
            "test/macro_f1": doc_metrics["macro_f1"],
            "test/weighted_f1": doc_metrics.get("weighted_f1"),
            "test/n": doc_metrics.get("n"),
            "best/test_accuracy": doc_metrics["accuracy"],
            "best/test_macro_f1": doc_metrics["macro_f1"],
            "best/test_weighted_f1": doc_metrics.get("weighted_f1"),
            "best/test_n": doc_metrics.get("n"),
        }
        if payload.get("best_val_macro_f1") is not None:
            best_log["best/val_macro_f1"] = payload["best_val_macro_f1"]
        if surface_metrics:
            best_log["surface/accuracy"] = surface_metrics["accuracy"]
            best_log["surface/macro_f1"] = surface_metrics["macro_f1"]
            best_log["surface/n"] = surface_metrics.get("n")
        if typed_slice:
            best_log["slice/typed_accuracy"] = typed_slice.get("accuracy")
            best_log["slice/typed_macro_f1"] = typed_slice.get("macro_f1")
            best_log["slice/typed_n"] = typed_slice.get("n")
        if ocr_slice:
            best_log["slice/ocr_accuracy"] = ocr_slice.get("accuracy")
            best_log["slice/ocr_macro_f1"] = ocr_slice.get("macro_f1")
            best_log["slice/ocr_n"] = ocr_slice.get("n")
        for key in (
            "mean_confidence",
            "mean_confidence_correct",
            "mean_confidence_incorrect",
            "ece",
        ):
            if key in conf and isinstance(conf[key], (int, float)):
                best_log[f"confidence/{key}"] = conf[key]
        wb.log({k: v for k, v in best_log.items() if v is not None}, step=step)
        step += 1

        # ---- Layer 3: per-class metrics (one class per step) ----
        per_class_rows = _per_class_rows(doc_metrics)
        for row in per_class_rows:
            label, precision, recall, f1, support = row
            wb.log(
                {
                    "layer": 3,
                    "best/class_precision": precision,
                    "best/class_recall": recall,
                    "best/class_f1": f1,
                    "best/class_support": support,
                },
                step=step,
            )
            step += 1
        if per_class_rows:
            wb.log_table(
                "best/per_class",
                ["label", "precision", "recall", "f1", "support"],
                per_class_rows,
            )

        if surface_metrics:
            surface_rows = _per_class_rows(surface_metrics)
            for row in surface_rows:
                _label, precision, recall, f1, support = row
                wb.log(
                    {
                        "layer": 3,
                        "surface/class_precision": precision,
                        "surface/class_recall": recall,
                        "surface/class_f1": f1,
                        "surface/class_support": support,
                    },
                    step=step,
                )
                step += 1
            if surface_rows:
                wb.log_table(
                    "surface/per_class",
                    ["label", "precision", "recall", "f1", "support"],
                    surface_rows,
                )

        for surface_name, key_prefix in (
            (SURFACE_TYPED, "slice/typed"),
            (SURFACE_HANDWRITING_OCR, "slice/ocr"),
        ):
            sm = slice_metrics.get(surface_name)
            if not sm:
                continue
            rows = _per_class_rows(sm)
            for row in rows:
                _label, precision, recall, f1, support = row
                wb.log(
                    {
                        "layer": 3,
                        f"{key_prefix}/class_precision": precision,
                        f"{key_prefix}/class_recall": recall,
                        f"{key_prefix}/class_f1": f1,
                        f"{key_prefix}/class_support": support,
                    },
                    step=step,
                )
                step += 1
            if rows:
                wb.log_table(
                    f"{key_prefix}/per_class",
                    ["label", "precision", "recall", "f1", "support"],
                    rows,
                )

        # ---- Layer 4: confidence bins (one bin per step) ----
        bins = conf.get("bins") or []
        for b in bins:
            wb.log(
                {
                    "layer": 4,
                    "confidence/bin": b.get("bin"),
                    "confidence/bin_n": b.get("n"),
                    "confidence/bin_accuracy": b.get("accuracy"),
                    "confidence/bin_mean_confidence": b.get("mean_confidence"),
                    "confidence/bin_lo": b.get("lo"),
                    "confidence/bin_hi": b.get("hi"),
                },
                step=step,
            )
            step += 1
        if bins:
            wb.log_table(
                "confidence/bins",
                ["bin", "lo", "hi", "n", "accuracy", "mean_confidence"],
                [
                    [
                        b.get("bin"),
                        b.get("lo"),
                        b.get("hi"),
                        b.get("n"),
                        b.get("accuracy"),
                        b.get("mean_confidence"),
                    ]
                    for b in bins
                ],
            )

        # ---- Layer 5: feature importances + confusion pairs ----
        features = payload.get("top_features") or []
        for rank, feat in enumerate(features):
            imp = feat.get("importance")
            if isinstance(imp, (int, float)):
                wb.log(
                    {
                        "layer": 5,
                        "interp/feature_rank": rank,
                        "interp/feature_importance": float(imp),
                    },
                    step=step,
                )
                step += 1
        if features:
            wb.log_table(
                "interp/top_features",
                ["feature", "importance"],
                [[f.get("feature"), f.get("importance")] for f in features],
            )

        pairs = payload.get("confusion_pairs") or top_confusion_pairs(doc_metrics)
        for rank, pair in enumerate(pairs):
            wb.log(
                {
                    "layer": 5,
                    "interp/confusion_rank": rank,
                    "interp/confusion_count": pair.get("count"),
                },
                step=step,
            )
            step += 1
        if pairs:
            wb.log_table(
                "interp/top_confusion_pairs",
                ["true", "pred", "count"],
                [[p["true"], p["pred"], p["count"]] for p in pairs],
            )

        if y_true is not None and preds and labels:
            y_true_ids = _label_ids(labels, list(y_true))
            y_pred_ids = _label_ids(labels, preds)
            if y_true_ids is not None and y_pred_ids is not None:
                wb.log_confusion_matrix(
                    key="best/confusion_matrix",
                    y_true=y_true_ids,
                    y_pred=y_pred_ids,
                    class_names=labels,
                )

        # Final marker so charts show total span
        wb.log({"layer": 99, "wandb/logged_steps": step}, step=step)

        # Summary panel (final values) — set after all steps so UI headlines match.
        summary = {k: v for k, v in best_log.items() if k != "layer" and v is not None}
        if payload.get("best_preset"):
            summary["best/preset"] = payload["best_preset"]
        for key in ("n_rows", "n_classes", "n_typed", "n_ocr", "typed_frac", "ocr_frac"):
            if key in data_profile:
                summary[f"data/{key}"] = data_profile[key]
        summary["wandb/logged_steps"] = step
        wb.summary(summary)

        if artifact_paths:
            try:
                wb.log_artifact_files(
                    name=f"random-forest-eval-{name}",
                    paths=artifact_paths,
                    artifact_type="evaluation",
                    metadata={
                        "accuracy": doc_metrics["accuracy"],
                        "macro_f1": doc_metrics["macro_f1"],
                        "best_preset": payload.get("best_preset"),
                        "logged_steps": step,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — artifacts are best-effort
                import logging

                logging.getLogger(__name__).warning(
                    "WandB artifact upload skipped: %s", exc
                )


def save_random_forest_bundle(
    model: Pipeline,
    out_dir: Path,
    metrics: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "random_forest_pipeline.joblib"
    joblib.dump(model, model_path)
    # JSON-friendly metrics (drop raw prediction lists if huge — keep summary)
    summary = {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "n": metrics.get("n"),
        "labels": metrics.get("labels"),
        "classification_report": metrics.get("classification_report"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }
    write_json(out_dir / "eval_metrics.json", summary)
    write_json(out_dir / "train_meta.json", meta or {})
    label2id = {lab: i for i, lab in enumerate(metrics.get("labels") or list(model.classes_))}
    write_json(out_dir / "label2id.json", label2id)
    return model_path


def write_predictions_jsonl(
    path: Path,
    record_ids: list[Any],
    y_true: list[Any],
    y_pred: list[Any],
    surfaces: list[Any] | None = None,
    confidences: list[float] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, rid in enumerate(record_ids):
        row = {
            "record_id": rid,
            "true_label": y_true[i],
            "predicted_label": y_pred[i],
            "correct": y_true[i] == y_pred[i],
        }
        if surfaces is not None:
            row["surface"] = surfaces[i]
        if confidences is not None:
            row["confidence"] = float(confidences[i])
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
