"""
Stage 1 — Document Processing (paper §V.C / Fig. 1).

Convert PDF files / document images into page-level images, enhance to
~300 DPI grayscale while retaining dimensions, and run OCR (PyTesseract
when available) to produce text + bounding boxes for downstream stages.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from src.docie.types import OcrWord, PageImage, ProcessedDocument

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_SUFFIXES = {".pdf"}
DEFAULT_DPI = 300


def _safe_id(record_id: str) -> str:
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:16]
    readable = re.sub(r"[^\w.+-]+", "_", record_id)[:80].strip("._") or "record"
    return f"{readable}__{digest}"


def _preprocess_image(
    image: Image.Image,
    *,
    grayscale: bool = True,
    target_dpi: int = DEFAULT_DPI,
) -> tuple[Image.Image, int]:
    """
    Paper pre-processing: enhance to 300 dpi, convert to grayscale, keep
    original pixel dimensions (no forced resize / crop).
    """
    img = image.convert("RGB")
    if grayscale:
        img = ImageOps.grayscale(img).convert("RGB")
    # Preserve native resolution; record the target dpi used for PDF renders.
    return img, target_dpi


def _ocr_page(image: Image.Image) -> tuple[str, list[OcrWord], str]:
    """OCR with pytesseract when installed; otherwise empty text."""
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return "", [], "none"

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        logger.warning("pytesseract OCR failed: %s", exc)
        try:
            text = pytesseract.image_to_string(image) or ""
            return text.strip(), [], "pytesseract_string"
        except Exception:
            return "", [], "pytesseract_failed"

    w, h = image.size
    words: list[OcrWord] = []
    lines: dict[tuple[int, int], list[str]] = {}
    n = len(data.get("text") or [])
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        x, y, bw, bh = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        # LayoutLM-style 0–1000 normalized boxes
        bbox = [
            max(0, min(1000, int(1000 * x / max(w, 1)))),
            max(0, min(1000, int(1000 * y / max(h, 1)))),
            max(0, min(1000, int(1000 * (x + bw) / max(w, 1)))),
            max(0, min(1000, int(1000 * (y + bh) / max(h, 1)))),
        ]
        words.append(OcrWord(text=raw, bbox=bbox, conf=conf / 100.0))
        key = (int(data.get("block_num", [0])[i]), int(data.get("line_num", [0])[i]))
        lines.setdefault(key, []).append(raw)

    text = "\n".join(" ".join(parts) for parts in lines.values()).strip()
    if not text and words:
        text = " ".join(w.text for w in words)
    return text, words, "pytesseract"


def _pdf_text_page(path: Path, page_index: int) -> str:
    """Native PDF text layer fallback when OCR is unavailable / empty."""
    try:
        import fitz  # pymupdf
    except ImportError:
        return ""
    try:
        doc = fitz.open(path)
        if page_index >= len(doc):
            doc.close()
            return ""
        text = (doc[page_index].get_text("text") or "").strip()
        doc.close()
        return text
    except Exception as exc:
        logger.warning("PDF text extract failed for page %d: %s", page_index, exc)
        return ""


def _render_pdf_pages(
    path: Path,
    out_dir: Path,
    record_id: str,
    *,
    dpi: int = DEFAULT_DPI,
    grayscale: bool = True,
    run_ocr: bool = True,
) -> list[PageImage]:
    import fitz  # pymupdf

    doc = fitz.open(path)
    pages: list[PageImage] = []
    safe = _safe_id(record_id)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                img = img.convert("RGB")
            img, used_dpi = _preprocess_image(img, grayscale=grayscale, target_dpi=dpi)
            image_path = out_dir / f"{safe}__page_{i:03d}.png"
            img.save(image_path, format="PNG")

            text = ""
            words: list[OcrWord] = []
            ocr_backend = "none"
            if run_ocr:
                text, words, ocr_backend = _ocr_page(img)
            if not text.strip():
                fallback = _pdf_text_page(path, i)
                if fallback:
                    text = fallback
                    ocr_backend = (
                        f"{ocr_backend}+pymupdf_text"
                        if ocr_backend not in {"none", "pytesseract_failed"}
                        else "pymupdf_text"
                    )
                    if not words:
                        words = [
                            OcrWord(text=tok)
                            for tok in fallback.split()
                            if tok.strip()
                        ]

            pages.append(
                PageImage(
                    page_index=i,
                    image_path=image_path,
                    width=img.width,
                    height=img.height,
                    dpi=used_dpi,
                    grayscale=grayscale,
                    text=text,
                    words=words,
                    ocr_backend=ocr_backend,
                )
            )
    finally:
        doc.close()
    return pages


def _process_image_file(
    path: Path,
    out_dir: Path,
    record_id: str,
    *,
    dpi: int = DEFAULT_DPI,
    grayscale: bool = True,
    run_ocr: bool = True,
    fallback_text: str | None = None,
) -> list[PageImage]:
    img = Image.open(path).convert("RGB")
    img, used_dpi = _preprocess_image(img, grayscale=grayscale, target_dpi=dpi)
    safe = _safe_id(record_id)
    image_path = out_dir / f"{safe}__page_000.png"
    img.save(image_path, format="PNG")

    text = ""
    words: list[OcrWord] = []
    ocr_backend = "none"
    if run_ocr:
        text, words, ocr_backend = _ocr_page(img)
    if not text.strip() and fallback_text:
        text = fallback_text.strip()
        ocr_backend = f"{ocr_backend}+fallback_text" if ocr_backend != "none" else "fallback_text"
        if not words:
            words = [OcrWord(text=tok) for tok in text.split() if tok.strip()]

    return [
        PageImage(
            page_index=0,
            image_path=image_path,
            width=img.width,
            height=img.height,
            dpi=used_dpi,
            grayscale=grayscale,
            text=text,
            words=words,
            ocr_backend=ocr_backend,
        )
    ]


def _process_text_as_page(
    text: str,
    out_dir: Path,
    record_id: str,
    *,
    dpi: int = DEFAULT_DPI,
    grayscale: bool = True,
) -> list[PageImage]:
    """Render plain text to a page image so the image-first chain stays intact."""
    from src.extraction.render_forms import render_page

    img, words_meta, _truncated = render_page(text)
    img, used_dpi = _preprocess_image(img, grayscale=grayscale, target_dpi=dpi)
    safe = _safe_id(record_id)
    image_path = out_dir / f"{safe}__page_000.png"
    img.save(image_path, format="PNG")
    w, h = img.size
    words: list[OcrWord] = []
    for meta in words_meta:
        tok = (meta.get("text") or "").strip()
        if not tok:
            continue
        x0, y0, x1, y1 = meta["bbox"]
        words.append(
            OcrWord(
                text=tok,
                bbox=[
                    max(0, min(1000, int(1000 * x0 / max(w, 1)))),
                    max(0, min(1000, int(1000 * y0 / max(h, 1)))),
                    max(0, min(1000, int(1000 * x1 / max(w, 1)))),
                    max(0, min(1000, int(1000 * y1 / max(h, 1)))),
                ],
            )
        )
    return [
        PageImage(
            page_index=0,
            image_path=image_path,
            width=img.width,
            height=img.height,
            dpi=used_dpi,
            grayscale=grayscale,
            text=text.strip(),
            words=words,
            ocr_backend="rendered_text",
        )
    ]


def process_document_input(
    *,
    record_id: str = "adhoc",
    pdf_path: str | Path | None = None,
    image_path: str | Path | None = None,
    source_path: str | Path | None = None,
    text: str | None = None,
    cache_dir: Path,
    application: str = "salvage_claims",
    dpi: int = DEFAULT_DPI,
    grayscale: bool = True,
    run_ocr: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ProcessedDocument:
    """
    Fig. 1 Stage 1 entry point: ingest PDF / image / text → page images + OCR.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf = Path(pdf_path) if pdf_path else None
    image = Path(image_path) if image_path else None
    source = Path(source_path) if source_path else None

    if source and not pdf and not image:
        suffix = source.suffix.lower()
        if suffix in PDF_SUFFIXES:
            pdf = source
        elif suffix in IMAGE_SUFFIXES:
            image = source

    source_kind = "empty"
    pages: list[PageImage] = []
    resolved_source: str | None = None

    if pdf is not None:
        if not pdf.exists():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        pages = _render_pdf_pages(
            pdf,
            cache_dir,
            record_id,
            dpi=dpi,
            grayscale=grayscale,
            run_ocr=run_ocr,
        )
        source_kind = "pdf"
        resolved_source = str(pdf)
    elif image is not None:
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        pages = _process_image_file(
            image,
            cache_dir,
            record_id,
            dpi=dpi,
            grayscale=grayscale,
            run_ocr=run_ocr,
            fallback_text=text,
        )
        source_kind = "image"
        resolved_source = str(image)
    elif text and text.strip():
        pages = _process_text_as_page(
            text,
            cache_dir,
            record_id,
            dpi=dpi,
            grayscale=grayscale,
        )
        source_kind = "text"
        resolved_source = None
    else:
        source_kind = "empty"

    return ProcessedDocument(
        record_id=record_id,
        source_path=resolved_source,
        source_kind=source_kind,
        pages=pages,
        application=application,
        metadata=dict(metadata or {}),
    )
