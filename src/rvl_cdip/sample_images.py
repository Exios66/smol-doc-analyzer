"""Materialize RVL-CDIP sample images for the recreation / OpenRouter eval PoC.

Resolution order for each sample:
  1. Existing ``image_abspath`` when the file is present
  2. ``.venv/rvl_cdip/source/images/{image_relpath}``
  3. Extract the single member from ``rvl-cdip.tar.gz`` (when the archive exists)

OCR text is optional and written beside the image as ``*.ocr.txt``.
"""

from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.rvl_cdip.paths import archive_path, assert_path_under_venv, images_dir

logger = logging.getLogger(__name__)


@dataclass
class MaterializeResult:
    document_id: str
    image_relpath: str
    image_path: Path | None
    extracted: bool = False
    ocr_path: Path | None = None
    ocr_text: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "image_relpath": self.image_relpath,
            "image_path": str(self.image_path) if self.image_path else None,
            "extracted": self.extracted,
            "ocr_path": str(self.ocr_path) if self.ocr_path else None,
            "ocr_chars": len(self.ocr_text or ""),
            "error": self.error,
        }


def _candidate_paths(sample: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    abspath = sample.get("image_abspath")
    if abspath:
        out.append(Path(str(abspath)))
    rel = str(sample.get("image_relpath") or "").lstrip("/")
    if rel:
        out.append(images_dir() / rel)
    return out


def resolve_image_path(sample: dict[str, Any]) -> Path | None:
    """Return a local image path if already on disk."""
    for path in _candidate_paths(sample):
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path.resolve()
        except OSError:
            continue
    return None


def _tar_member_names(tf: tarfile.TarFile, relpath: str) -> list[str]:
    """Possible archive member names for a dataset-relative image path."""
    rel = relpath.lstrip("/")
    names = [rel, f"./{rel}", f"images/{Path(rel).name}"]
    # Some archives nest under a top-level folder.
    basenames = {Path(rel).name}
    matches: list[str] = []
    members = tf.getnames()
    for name in members:
        norm = name.lstrip("./")
        if (
            norm == rel
            or norm.endswith("/" + rel)
            or (Path(norm).name in basenames and norm.endswith(rel))
        ):
            matches.append(name)
        elif name in names:
            matches.append(name)
    # Prefer exact suffix match on image_relpath
    exact = [m for m in members if m.lstrip("./").endswith(rel)]
    if exact:
        return exact
    return matches


def extract_image_from_archive(
    image_relpath: str,
    *,
    archive: Path | None = None,
    dest_root: Path | None = None,
) -> Path | None:
    """Extract one image member from ``rvl-cdip.tar.gz`` into ``images_dir``."""
    arch = archive or archive_path()
    if not arch.is_file():
        return None
    dest_root = assert_path_under_venv(dest_root or images_dir())
    dest = dest_root / image_relpath.lstrip("/")
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.resolve()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arch, "r:*") as tf:
        candidates = _tar_member_names(tf, image_relpath)
        if not candidates:
            logger.warning("Archive member not found for %s", image_relpath)
            return None
        member_name = candidates[0]
        member = tf.getmember(member_name)
        if not member.isfile():
            return None
        extracted = tf.extractfile(member)
        if extracted is None:
            return None
        data = extracted.read()
        dest.write_bytes(data)
        assert_path_under_venv(dest)
        return dest.resolve()


def materialize_sample_image(sample: dict[str, Any]) -> MaterializeResult:
    """Resolve or extract the image for one recreation-sample row."""
    doc_id = str(sample.get("document_id") or "")
    rel = str(sample.get("image_relpath") or "")
    existing = resolve_image_path(sample)
    if existing is not None:
        return MaterializeResult(
            document_id=doc_id,
            image_relpath=rel,
            image_path=existing,
            extracted=False,
        )
    if not rel:
        return MaterializeResult(
            document_id=doc_id,
            image_relpath=rel,
            image_path=None,
            error="missing image_relpath",
        )
    try:
        extracted = extract_image_from_archive(rel)
    except Exception as exc:  # noqa: BLE001 — surface as row error
        return MaterializeResult(
            document_id=doc_id,
            image_relpath=rel,
            image_path=None,
            error=f"extract failed: {exc}",
        )
    if extracted is None:
        arch = archive_path()
        hint = (
            f"image not on disk and archive missing ({arch})"
            if not arch.is_file()
            else f"member not found in archive for {rel}"
        )
        return MaterializeResult(
            document_id=doc_id,
            image_relpath=rel,
            image_path=None,
            error=hint,
        )
    return MaterializeResult(
        document_id=doc_id,
        image_relpath=rel,
        image_path=extracted,
        extracted=True,
    )


def extract_images_batch(
    image_relpaths: Sequence[str],
    *,
    archive: Path | None = None,
    dest_root: Path | None = None,
) -> dict[str, Path]:
    """Extract many members from ``rvl-cdip.tar.gz`` in a single sequential pass.

    Avoids re-scanning the ~38 GB gzipped archive once per image (which is what
    calling :func:`extract_image_from_archive` in a loop does via ``getnames()``).
    """
    arch = archive or archive_path()
    if not arch.is_file():
        return {}
    dest_root = assert_path_under_venv(dest_root or images_dir())

    needed: dict[str, Path] = {}
    for rel in image_relpaths:
        rel_norm = str(rel).lstrip("/")
        if not rel_norm:
            continue
        dest = dest_root / rel_norm
        if dest.is_file() and dest.stat().st_size > 0:
            needed[rel_norm] = dest.resolve()
        else:
            needed.setdefault(rel_norm, dest)  # Path placeholder until extracted

    pending = {rel for rel, path in needed.items() if not path.is_file()}
    if not pending:
        return {rel: path for rel, path in needed.items() if path.is_file()}

    # Suffix index: archive members may be prefixed (e.g. images/...).
    pending_by_suffix = {rel: rel for rel in pending}
    found: dict[str, Path] = {
        rel: path for rel, path in needed.items() if path.is_file()
    }

    logger.info(
        "Batch-extracting %d images from %s (single pass)",
        len(pending),
        arch,
    )
    # Fast lookup: exact member name, images/<rel>, or */<rel>
    pending_lookup: dict[str, str] = {}
    for rel in pending:
        pending_lookup[rel] = rel
        pending_lookup[f"images/{rel}"] = rel

    with tarfile.open(arch, "r:*") as tf:
        for member in tf:
            if not member.isfile() or not pending_lookup:
                if not pending_lookup:
                    break
                continue
            norm = member.name.lstrip("./")
            match_rel = pending_lookup.get(norm)
            if match_rel is None:
                for rel in list(pending):
                    if norm.endswith("/" + rel):
                        match_rel = rel
                        break
            if match_rel is None or match_rel not in pending:
                continue
            dest = dest_root / match_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            dest.write_bytes(extracted.read())
            found[match_rel] = dest.resolve()
            pending.discard(match_rel)
            pending_lookup.pop(match_rel, None)
            pending_lookup.pop(f"images/{match_rel}", None)

    pending_by_suffix = {rel: rel for rel in pending}

    if pending_by_suffix:
        logger.warning(
            "Archive batch extract missing %d/%d members (e.g. %s)",
            len(pending_by_suffix),
            len(pending),
            next(iter(pending_by_suffix)),
        )
    return found


def materialize_samples(
    samples: Sequence[dict[str, Any]],
    *,
    run_ocr: bool = False,
) -> list[MaterializeResult]:
    """Materialize images (and optional OCR) for a batch of samples."""
    # Resolve what is already on disk; batch-extract the rest in one tar pass.
    prelim: list[MaterializeResult | None] = [None] * len(samples)
    missing_rels: list[str] = []
    for idx, sample in enumerate(samples):
        doc_id = str(sample.get("document_id") or "")
        rel = str(sample.get("image_relpath") or "")
        existing = resolve_image_path(sample)
        if existing is not None:
            prelim[idx] = MaterializeResult(
                document_id=doc_id,
                image_relpath=rel,
                image_path=existing,
                extracted=False,
            )
        elif rel:
            missing_rels.append(rel.lstrip("/"))
        else:
            prelim[idx] = MaterializeResult(
                document_id=doc_id,
                image_relpath=rel,
                image_path=None,
                error="missing image_relpath",
            )

    extracted_map = extract_images_batch(missing_rels) if missing_rels else {}
    arch = archive_path()

    results: list[MaterializeResult] = []
    for idx, sample in enumerate(samples):
        if prelim[idx] is not None:
            result = prelim[idx]
            assert result is not None
        else:
            doc_id = str(sample.get("document_id") or "")
            rel = str(sample.get("image_relpath") or "").lstrip("/")
            path = extracted_map.get(rel)
            if path is None:
                hint = (
                    f"image not on disk and archive missing ({arch})"
                    if not arch.is_file()
                    else f"member not found in archive for {rel}"
                )
                result = MaterializeResult(
                    document_id=doc_id,
                    image_relpath=rel,
                    image_path=None,
                    error=hint,
                )
            else:
                result = MaterializeResult(
                    document_id=doc_id,
                    image_relpath=rel,
                    image_path=path,
                    extracted=True,
                )

        if run_ocr and result.image_path is not None:
            ocr_path, ocr_text, ocr_err = ocr_image(result.image_path)
            result.ocr_path = ocr_path
            result.ocr_text = ocr_text
            if ocr_err and result.error is None:
                result.error = ocr_err
        results.append(result)
    return results


def ocr_image(image_path: Path) -> tuple[Path | None, str | None, str | None]:
    """Run pytesseract OCR; cache to ``<image>.ocr.txt`` beside the file."""
    out = Path(str(image_path) + ".ocr.txt")
    if out.is_file() and out.stat().st_size > 0:
        text = out.read_text(encoding="utf-8", errors="replace")
        return out, text, None
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None, None, "OCR requires pillow + pytesseract (pip install -e '.[ocr]')"

    try:
        with Image.open(image_path) as im:
            text = pytesseract.image_to_string(im.convert("RGB"))
        out.write_text(text, encoding="utf-8")
        return out, text, None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"OCR failed: {exc}"


def summarize_materialize(results: Iterable[MaterializeResult]) -> dict[str, Any]:
    rows = list(results)
    ok = [r for r in rows if r.image_path is not None]
    return {
        "n_samples": len(rows),
        "n_with_image": len(ok),
        "n_extracted": sum(1 for r in ok if r.extracted),
        "n_with_ocr": sum(1 for r in rows if r.ocr_text),
        "n_errors": sum(1 for r in rows if r.error),
        "archive_present": archive_path().is_file(),
        "images_dir": str(images_dir()),
    }
