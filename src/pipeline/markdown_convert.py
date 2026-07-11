"""
Convert PNG / PDF / plain text into compact structured markdown.

Markdown is fed to downstream LLM stages instead of raw page images so
token use stays low while layout cues (headings, key/value fields, lists)
remain explicit for contextual understanding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.extraction.render_forms import FIELD_PATTERNS

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class MarkdownConversion:
    markdown: str
    plain_text: str
    source_kind: str  # text | png | pdf | image | empty
    backend: str
    pages: int = 1
    char_count: int = 0
    approx_tokens: int = 0
    source_path: str | None = None
    extras: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "plain_text": self.plain_text,
            "source_kind": self.source_kind,
            "backend": self.backend,
            "pages": self.pages,
            "char_count": self.char_count,
            "approx_tokens": self.approx_tokens,
            "source_path": self.source_path,
            "extras": self.extras or {},
        }


def approx_token_count(text: str) -> int:
    """Rough token estimate (~4 chars/token) for logging / comparison."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_to_structured_markdown(text: str, title: str | None = None) -> str:
    """
    Turn form-like plain text into compact markdown.

    - First non-empty line becomes an H1 when it looks like a title
    - `Label: value` lines become a definition-style field list
    - Remaining blocks become paragraphs / bullet lists
    """
    text = _normalize_whitespace(text)
    if not text:
        return ""

    lines = text.split("\n")
    md_lines: list[str] = []
    if title:
        md_lines.append(f"# {title.strip()}")
        md_lines.append("")

    # Known field prefixes for insurance forms
    prefix_map = {prefix.lower(): field for field, prefix in FIELD_PATTERNS}

    field_rows: list[tuple[str, str]] = []
    body_blocks: list[str] = []
    title_taken = bool(title)
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue

        # Title heuristic: short ALL-CAPS / form header line
        if not title_taken and i < 3 and (
            stripped.isupper() or re.match(r"^[A-Z][A-Za-z0-9 /-]{3,60}$", stripped)
        ):
            md_lines.append(f"# {stripped.title() if stripped.isupper() else stripped}")
            md_lines.append("")
            title_taken = True
            i += 1
            continue

        # Section headers (short lines without colon, mostly words)
        if (
            ":" not in stripped
            and len(stripped) <= 48
            and not stripped.endswith(".")
            and re.match(r"^[A-Za-z][A-Za-z0-9 /&-]{2,}$", stripped)
        ):
            # Flush pending fields before a new section
            if field_rows:
                md_lines.extend(_fields_to_md(field_rows))
                md_lines.append("")
                field_rows = []
            md_lines.append(f"## {stripped}")
            md_lines.append("")
            i += 1
            continue

        # Key: value
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key_s, val_s = key.strip(), val.strip()
            lower_key = key_s.lower()
            # Map to canonical field names when possible
            canon = None
            for prefix, field in prefix_map.items():
                if lower_key == prefix.rstrip(":").lower() or lower_key == field.replace("_", " "):
                    canon = field
                    break
            label = canon.replace("_", " ").title() if canon else key_s
            if val_s:
                field_rows.append((label, val_s))
            else:
                # value may continue on following lines — keep as section-ish
                field_rows.append((label, ""))
            i += 1
            continue

        # Bullet-ish
        if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]\s+", stripped):
            if field_rows:
                md_lines.extend(_fields_to_md(field_rows))
                md_lines.append("")
                field_rows = []
            bullet = re.sub(r"^[-*•]\s*", "", stripped)
            bullet = re.sub(r"^\d+[.)]\s*", "", bullet)
            body_blocks.append(f"- {bullet}")
            i += 1
            continue

        if field_rows:
            md_lines.extend(_fields_to_md(field_rows))
            md_lines.append("")
            field_rows = []
        body_blocks.append(stripped)
        i += 1

    if field_rows:
        md_lines.extend(_fields_to_md(field_rows))
        md_lines.append("")
    if body_blocks:
        # Collapse consecutive non-bullets into paragraphs
        para: list[str] = []
        for block in body_blocks:
            if block.startswith("- "):
                if para:
                    md_lines.append(" ".join(para))
                    md_lines.append("")
                    para = []
                md_lines.append(block)
            else:
                para.append(block)
        if para:
            md_lines.append(" ".join(para))
            md_lines.append("")

    return _normalize_whitespace("\n".join(md_lines)) + "\n"


def _fields_to_md(rows: list[tuple[str, str]]) -> list[str]:
    out = ["| Field | Value |", "| --- | --- |"]
    for label, value in rows:
        safe_label = label.replace("|", "\\|")
        safe_val = (value or "").replace("|", "\\|")
        out.append(f"| **{safe_label}** | {safe_val} |")
    return out


def markdown_to_plain(markdown: str) -> str:
    """Strip light markdown syntax for encoder models that expect plain text."""
    text = markdown
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^\| Field \| Value \|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\| --- \| --- \|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|\s*\*\*([^*]+)\*\*\s*\|\s*(.*?)\s*\|$", r"\1: \2", text, flags=re.MULTILINE)
    text = re.sub(r"^\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|$", r"\1: \2", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    return _normalize_whitespace(text)


def _ocr_image(path: Path) -> tuple[str, str]:
    """OCR an image with pytesseract when available; else empty."""
    try:
        from PIL import Image
        import pytesseract  # type: ignore
    except Exception as exc:
        logger.info("OCR unavailable for %s (%s)", path, exc)
        return "", "none"

    try:
        image = Image.open(path).convert("RGB")
        text = pytesseract.image_to_string(image) or ""
        return _normalize_whitespace(text), "pytesseract"
    except Exception as exc:
        logger.warning("OCR failed for %s (%s)", path, exc)
        return "", "pytesseract_failed"


def png_to_markdown(path: Path, fallback_text: str | None = None) -> MarkdownConversion:
    path = Path(path)
    ocr_text, ocr_backend = _ocr_image(path)
    source_text = ocr_text or (fallback_text or "")
    backend = ocr_backend if ocr_text else ("fallback_text" if fallback_text else "empty")
    if not source_text:
        # Last resort: note the image path so downstream stages still have context
        md = f"# Document Image\n\n_Source:_ `{path.name}`\n\n_(No text extracted from image.)_\n"
        return MarkdownConversion(
            markdown=md,
            plain_text="",
            source_kind="png",
            backend=backend,
            pages=1,
            char_count=len(md),
            approx_tokens=approx_token_count(md),
            source_path=str(path),
        )
    md = text_to_structured_markdown(source_text, title=None)
    return MarkdownConversion(
        markdown=md,
        plain_text=markdown_to_plain(md),
        source_kind="png",
        backend=backend,
        pages=1,
        char_count=len(md),
        approx_tokens=approx_token_count(md),
        source_path=str(path),
        extras={"ocr_chars": len(ocr_text), "used_fallback_text": bool(fallback_text and not ocr_text)},
    )


def _pdf_extract_pymupdf(path: Path) -> tuple[str, int, str]:
    import fitz  # pymupdf

    doc = fitz.open(path)
    parts: list[str] = []
    for i, page in enumerate(doc):
        page_text = page.get_text("text") or ""
        if page_text.strip():
            parts.append(f"<!-- page {i + 1} -->\n{page_text.strip()}")
    n = doc.page_count
    doc.close()
    return "\n\n".join(parts), n, "pymupdf"


def _pdf_extract_pypdf(path: Path) -> tuple[str, int, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(f"<!-- page {i + 1} -->\n{page_text.strip()}")
    return "\n\n".join(parts), len(reader.pages), "pypdf"


def pdf_to_markdown(path: Path, fallback_text: str | None = None) -> MarkdownConversion:
    path = Path(path)
    raw = ""
    pages = 1
    backend = "none"
    try:
        raw, pages, backend = _pdf_extract_pymupdf(path)
    except Exception as exc_a:
        try:
            raw, pages, backend = _pdf_extract_pypdf(path)
        except Exception as exc_b:
            logger.warning("PDF text extract failed (%s / %s)", exc_a, exc_b)
            raw, pages, backend = (fallback_text or ""), 1, "fallback_text" if fallback_text else "failed"

    if not raw.strip() and fallback_text:
        raw = fallback_text
        backend = f"{backend}+fallback_text"

    # Drop page markers before structuring, keep page count metadata
    cleaned = re.sub(r"<!-- page \d+ -->\n?", "", raw)
    md = text_to_structured_markdown(cleaned) if cleaned.strip() else (
        f"# PDF Document\n\n_Source:_ `{path.name}`\n\n_(No text extracted.)_\n"
    )
    # Re-insert page separators lightly when multi-page
    if pages > 1 and cleaned.strip():
        page_chunks = re.split(r"<!-- page \d+ -->\n?", raw)
        page_chunks = [c.strip() for c in page_chunks if c.strip()]
        if len(page_chunks) > 1:
            md_parts = [
                text_to_structured_markdown(chunk, title=f"Page {idx}")
                for idx, chunk in enumerate(page_chunks, start=1)
            ]
            md = "\n\n".join(md_parts)

    return MarkdownConversion(
        markdown=md if md.endswith("\n") else md + "\n",
        plain_text=markdown_to_plain(md),
        source_kind="pdf",
        backend=backend,
        pages=pages,
        char_count=len(md),
        approx_tokens=approx_token_count(md),
        source_path=str(path),
    )


def convert_to_markdown(
    *,
    text: str | None = None,
    image_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    source_path: str | Path | None = None,
) -> MarkdownConversion:
    """
    Convert the best available source into structured markdown.

    Priority: explicit pdf_path → image_path → source_path suffix → text.
    """
    # Resolve generic source_path
    if source_path and not pdf_path and not image_path:
        sp = Path(source_path)
        suffix = sp.suffix.lower()
        if suffix in PDF_SUFFIXES:
            pdf_path = sp
        elif suffix in IMAGE_SUFFIXES:
            image_path = sp

    if pdf_path:
        return pdf_to_markdown(Path(pdf_path), fallback_text=text)
    if image_path:
        return png_to_markdown(Path(image_path), fallback_text=text)
    if text and text.strip():
        md = text_to_structured_markdown(text)
        return MarkdownConversion(
            markdown=md if md.endswith("\n") else md + "\n",
            plain_text=markdown_to_plain(md),
            source_kind="text",
            backend="structured_text",
            pages=1,
            char_count=len(md),
            approx_tokens=approx_token_count(md),
            source_path=None,
        )
    empty = "# Empty Document\n\n_(No content provided.)_\n"
    return MarkdownConversion(
        markdown=empty,
        plain_text="",
        source_kind="empty",
        backend="empty",
        pages=0,
        char_count=len(empty),
        approx_tokens=approx_token_count(empty),
    )
