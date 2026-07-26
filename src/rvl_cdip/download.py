"""Download RVL-CDIP artifacts exclusively into ``.venv/rvl_cdip/``.

By default only the small split label files (~17 MB total) are fetched. The
~38 GB ``rvl-cdip.tar.gz`` image archive requires an explicit opt-in flag.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.rvl_cdip.paths import (
    HF_DATASET_ID,
    IMAGE_ARCHIVE_BYTES_ESTIMATE,
    IMAGE_ARCHIVE_REMOTE,
    IMAGE_DOWNLOAD_MIN_FREE_BYTES,
    LABEL_FILES,
    apply_hf_cache_env,
    archive_path,
    assert_path_under_venv,
    hf_home,
    images_dir,
    labels_dir,
    rvl_root,
    source_dir,
)

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    kind: str
    remote_ref: str
    local_path: Path
    bytes: int | None = None
    skipped: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


def _hf_token() -> str | None:
    try:
        from src.utils.config import Config

        token = Config.load().hf_token
        return token or None
    except Exception:
        return None


def _hub_download(repo_file: str, *, local_dir: Path) -> Path:
    """Download one file from the Hub dataset into ``local_dir`` (under .venv)."""
    apply_hf_cache_env()
    assert_path_under_venv(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    token = _hf_token()
    # Prefer aharley/rvl_cdip; fall back to the legacy rvl_cdip resolve path
    # used by the upstream dataset script if the namespaced copy is missing.
    last_err: Exception | None = None
    for repo_id in (HF_DATASET_ID, "rvl_cdip"):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=token,
            )
            out = Path(path)
            assert_path_under_venv(out)
            return out
        except Exception as exc:  # noqa: BLE001 — try alternate repo id
            last_err = exc
            logger.debug("Hub download via %s failed for %s: %s", repo_id, repo_file, exc)
    raise RuntimeError(
        f"Failed to download {repo_file} from Hugging Face "
        f"({HF_DATASET_ID} / rvl_cdip): {last_err}"
    )


def free_bytes(path: Path | None = None) -> int:
    usage = shutil.disk_usage(path or rvl_root())
    return int(usage.free)


def download_labels(*, force: bool = False) -> list[DownloadResult]:
    """Fetch train/test/val label lists into ``.venv/rvl_cdip/source/labels``."""
    apply_hf_cache_env()
    out_dir = labels_dir()
    assert_path_under_venv(out_dir)
    results: list[DownloadResult] = []

    for split, remote in LABEL_FILES.items():
        dest_name = Path(remote).name  # train.txt / test.txt / val.txt
        dest = out_dir / dest_name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            results.append(
                DownloadResult(
                    kind="labels",
                    remote_ref=f"{HF_DATASET_ID}/{remote}",
                    local_path=dest,
                    bytes=dest.stat().st_size,
                    skipped=True,
                    detail={"split": split, "reason": "already_present"},
                )
            )
            continue

        # Download into a staging folder under source/, then copy the named
        # label file into labels/ so the layout stays predictable.
        staging = source_dir() / "_hub_labels"
        staging.mkdir(parents=True, exist_ok=True)
        downloaded = _hub_download(remote, local_dir=staging)
        # hf_hub_download with local_dir may nest as data/train.txt
        candidate = downloaded if downloaded.name == dest_name else staging / remote
        if not candidate.exists():
            # Search staging tree for the filename.
            matches = list(staging.rglob(dest_name))
            if not matches:
                raise FileNotFoundError(f"Downloaded label file not found: {dest_name}")
            candidate = matches[0]
        shutil.copy2(candidate, dest)
        assert_path_under_venv(dest)
        results.append(
            DownloadResult(
                kind="labels",
                remote_ref=f"{HF_DATASET_ID}/{remote}",
                local_path=dest,
                bytes=dest.stat().st_size,
                detail={"split": split},
            )
        )
        logger.info("Downloaded RVL-CDIP %s labels → %s (%s bytes)", split, dest, dest.stat().st_size)

    # Drop Hub local_dir staging copies; canonical labels live in labels/.
    staging = source_dir() / "_hub_labels"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    return results


def image_download_preflight(*, force: bool = False) -> dict[str, Any]:
    """Report paths + free space for the ~38 GB archive download (no network I/O).

    All write targets are under ``.venv/rvl_cdip/`` (never ``data/`` or ``~/.cache``).
    """
    apply_hf_cache_env()
    root = rvl_root()
    dest = archive_path()
    free = free_bytes(dest.parent)
    already = dest.is_file() and dest.stat().st_size > 0 and not force
    enough_space = free >= IMAGE_DOWNLOAD_MIN_FREE_BYTES
    return {
        "dataset_id": HF_DATASET_ID,
        "remote_ref": f"{HF_DATASET_ID}/{IMAGE_ARCHIVE_REMOTE}",
        "writes_only_under": str(root.resolve()),
        "archive_path": str(dest.resolve()),
        "source_dir": str(source_dir().resolve()),
        "images_dir": str(images_dir().resolve()),
        "hf_home": str(hf_home().resolve()),
        "archive_estimate_gb": round(IMAGE_ARCHIVE_BYTES_ESTIMATE / (1024**3), 2),
        "min_free_gb": round(IMAGE_DOWNLOAD_MIN_FREE_BYTES / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
        "free_bytes": free,
        "enough_free_space": enough_space,
        "archive_already_present": already,
        "confirmation_phrase": "writes only under .venv/rvl_cdip",
    }


def format_image_download_preflight(plan: dict[str, Any] | None = None) -> str:
    """Human-readable confirmation block for CLI / notebook."""
    plan = plan or image_download_preflight()
    lines = [
        "RVL-CDIP image archive download — preflight",
        "",
        "CONFIRMATION: all Hub cache + archive writes stay under:",
        f"  {plan['writes_only_under']}",
        "  (not data/, not ~/.cache/huggingface)",
        "",
        f"Remote:     {plan['remote_ref']}",
        f"Archive →   {plan['archive_path']}",
        f"HF home →   {plan['hf_home']}",
        f"Images dir: {plan['images_dir']}",
        "",
        f"Archive size (est.): {plan['archive_estimate_gb']} GB",
        f"Free space now:      {plan['free_gb']} GB "
        f"(need ≥ {plan['min_free_gb']} GB)",
        f"Enough free space:   {plan['enough_free_space']}",
        f"Already present:     {plan['archive_already_present']}",
        "",
        f"Ack phrase: \"{plan['confirmation_phrase']}\"",
    ]
    return "\n".join(lines)


def download_images(
    *,
    force: bool = False,
    i_understand_large_download: bool = False,
    confirm_writes_under_venv: bool = False,
) -> DownloadResult:
    """Download the ~38 GB image archive into ``.venv/rvl_cdip/source/``.

    Requires ``i_understand_large_download=True`` and
    ``confirm_writes_under_venv=True`` (explicit acknowledgement that all
    writes stay under ``.venv/rvl_cdip/``). Checks free disk space first.
    """
    if not i_understand_large_download:
        raise RuntimeError(
            "Refusing to download rvl-cdip.tar.gz (~38 GB). Re-run with "
            "--i-understand-large-download if you have enough disk space "
            f"(recommend ≥ {IMAGE_DOWNLOAD_MIN_FREE_BYTES // (1024**3)} GB free)."
        )
    if not confirm_writes_under_venv:
        raise RuntimeError(
            "Refusing to download without explicit confirmation that writes "
            "stay under .venv/rvl_cdip/. Re-run with --confirm-writes-under-venv "
            "(or call download_images(confirm_writes_under_venv=True)). "
            "Preview with: python -m src.rvl_cdip download-images --preflight"
        )

    apply_hf_cache_env()
    dest = archive_path()
    assert_path_under_venv(dest)
    assert_path_under_venv(hf_home())

    plan = image_download_preflight(force=force)
    if not plan["enough_free_space"] and not (
        dest.exists() and dest.stat().st_size > 0 and not force
    ):
        raise RuntimeError(
            f"Insufficient free disk space for RVL-CDIP images: "
            f"{plan['free_gb']:.1f} GB free, need ≥ {plan['min_free_gb']:.0f} GB "
            f"(archive ≈ {plan['archive_estimate_gb']:.1f} GB). "
            f"All writes would have gone under {plan['writes_only_under']}."
        )

    if dest.exists() and dest.stat().st_size > 0 and not force:
        return DownloadResult(
            kind="images",
            remote_ref=f"{HF_DATASET_ID}/{IMAGE_ARCHIVE_REMOTE}",
            local_path=dest,
            bytes=dest.stat().st_size,
            skipped=True,
            detail={
                "reason": "already_present",
                "writes_only_under": plan["writes_only_under"],
            },
        )

    free = int(plan["free_bytes"])

    logger.warning(
        "Downloading RVL-CDIP image archive (~%.1f GB) into %s — writes stay "
        "under .venv/rvl_cdip only; this may take a long time.",
        IMAGE_ARCHIVE_BYTES_ESTIMATE / (1024**3),
        dest.parent,
    )
    staging = source_dir() / "_hub_archive"
    staging.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    downloaded = _hub_download(IMAGE_ARCHIVE_REMOTE, local_dir=staging)
    candidate = downloaded
    if candidate.name != dest.name:
        matches = list(staging.rglob(dest.name))
        if not matches:
            raise FileNotFoundError(f"Downloaded archive not found: {dest.name}")
        candidate = matches[0]
    if candidate.resolve() != dest.resolve():
        shutil.move(str(candidate), str(dest))
    assert_path_under_venv(dest)
    elapsed = time.time() - t0
    size = dest.stat().st_size
    logger.info("Downloaded RVL-CDIP archive → %s (%s bytes, %.1fs)", dest, size, elapsed)
    return DownloadResult(
        kind="images",
        remote_ref=f"{HF_DATASET_ID}/{IMAGE_ARCHIVE_REMOTE}",
        local_path=dest,
        bytes=size,
        detail={
            "elapsed_s": elapsed,
            "free_bytes_before": free,
            "writes_only_under": plan["writes_only_under"],
        },
    )


def ensure_labels(*, force: bool = False) -> list[DownloadResult]:
    """Idempotent: download label files if missing."""
    return download_labels(force=force)


def label_file_paths() -> dict[str, Path]:
    """Map split name → local label file path (may not exist yet)."""
    base = labels_dir()
    return {
        "train": base / "train.txt",
        "test": base / "test.txt",
        "validation": base / "val.txt",
    }


def iter_label_rows(path: Path, *, split: str) -> Iterable[tuple[str, int, int]]:
    """Yield ``(image_relpath, label_id, source_line)`` from a label file.

    Upstream format per line: ``<relpath> <label_id>`` where relpath is relative
    to the archive ``images/`` prefix (see aharley/rvl_cdip dataset script).
    """
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed RVL-CDIP label line in {path}#{line_no} ({split}): {raw!r}"
                )
            relpath, label_s = parts
            yield relpath, int(label_s), line_no
