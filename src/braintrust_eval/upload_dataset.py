"""Upload the fixed-size sample set to a Braintrust dataset (with image attachments)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.braintrust_eval.dataset import DEFAULT_OUT_DIR, load_samples
from src.utils.config import Config


def _require_braintrust():
    try:
        from braintrust import Attachment, init_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            'braintrust is not installed. Run: pip install -e ".[braintrust]"'
        ) from exc
    return Attachment, init_dataset


def upload_dataset(
    *,
    samples: Sequence[dict[str, Any]] | None = None,
    project: str | None = None,
    dataset_name: str | None = None,
    org_name: str | None = None,
    cfg: Config | None = None,
    clear_existing: bool = False,
) -> dict[str, Any]:
    """Insert fixed-size sample rows into a Braintrust dataset.

    Each row:
      input: {document_id, label_id, image Attachment}
      expected: gold RVL label string
      metadata: paths / seed / size
    """
    Attachment, init_dataset = _require_braintrust()
    config = cfg or Config.load()
    if not config.braintrust_api_key:
        raise RuntimeError(
            "BRAINTRUST_API_KEY is not set. Copy .env.example → .env and paste the key."
        )

    project_name = project or config.braintrust_project
    ds_name = dataset_name or config.braintrust_dataset
    org = org_name if org_name is not None else (config.braintrust_org or None)

    rows = list(samples) if samples is not None else load_samples()
    dataset = init_dataset(
        project=project_name,
        name=ds_name,
        org_name=org,
        api_key=config.braintrust_api_key,
        description=(
            "RVL-CDIP fixed-size sampled pages (1024×1024 padded PNGs, "
            "10 per class × 16 classes) for vision classification evals."
        ),
    )

    if clear_existing:
        # Dataset.clear() exists on recent SDKs; fall back to insert-only.
        clearer = getattr(dataset, "clear", None)
        if callable(clearer):
            clearer()

    n = 0
    for row in rows:
        image_path = Path(str(row["fixed_image_path"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing fixed image: {image_path}")
        attachment = Attachment(
            data=str(image_path),
            filename=image_path.name,
            content_type="image/png",
        )
        dataset.insert(
            input={
                "document_id": row["document_id"],
                "label_id": int(row["label_id"]),
                "image": attachment,
            },
            expected=str(row["label"]),
            metadata={
                "label": row["label"],
                "label_id": int(row["label_id"]),
                "fixed_image_path": str(image_path),
                "fixed_image_relpath": row.get("fixed_image_relpath"),
                "size": row.get("size"),
                "sample_seed": row.get("sample_seed"),
                "placeholder": bool(row.get("placeholder")),
                "split": row.get("split"),
            },
            id=str(row["document_id"]),
        )
        n += 1

    flush = getattr(dataset, "flush", None)
    if callable(flush):
        flush()
    summarize = getattr(dataset, "summarize", None)
    summary = summarize() if callable(summarize) else None

    return {
        "project": project_name,
        "dataset": ds_name,
        "org": org,
        "n_inserted": n,
        "local_samples_dir": str(DEFAULT_OUT_DIR),
        "summary": summary,
    }
