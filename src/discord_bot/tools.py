"""Chloride agent tools that call the local document-analysis pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

from src.discord_bot.formatters import compact_analysis, format_discord_summary
from src.pipeline.orchestrator import analyze_document
from src.utils.config import REPO_ROOT, Config

if TYPE_CHECKING:
    from coral.agent import Deps
    from pydantic_ai import RunContext

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".jsonl"}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_TOOLS_REGISTERED = False


def _inbox_dir() -> Path:
    cfg = Config.load()
    path = cfg.pipeline_cache_dir / "discord_inbox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(name: str) -> str:
    base = Path(name or "attachment.bin").name
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "attachment.bin"
    return cleaned[:180]


def _guess_kind(path: Path, content_type: str | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES or (content_type and "pdf" in content_type):
        return "pdf"
    if suffix in IMAGE_SUFFIXES or (content_type and content_type.startswith("image/")):
        return "image"
    if suffix in TEXT_SUFFIXES or (content_type and content_type.startswith("text/")):
        return "text"
    return "unknown"


async def _download_url(url: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("file://"):
        src = Path(url.removeprefix("file://"))
        dest.write_bytes(src.read_bytes())
        return dest
    if not url.startswith(("http://", "https://")):
        src = Path(url)
        dest.write_bytes(src.read_bytes())
        return dest

    def _fetch() -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "smol-doc-analyzer-discord-bot/0.1"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return resp.read()

    try:
        import asyncio

        data = await asyncio.to_thread(_fetch)
    except Exception:
        data = _fetch()
    dest.write_bytes(data)
    return dest


async def _attachment_from_message(message, attachment_index: int = 0) -> tuple[Path, str]:
    if message is None or not getattr(message, "attachments", None):
        raise ValueError("No Discord message attachments available.")
    if attachment_index < 0 or attachment_index >= len(message.attachments):
        raise ValueError(
            f"attachment_index={attachment_index} out of range "
            f"(message has {len(message.attachments)} attachment(s))."
        )
    att = message.attachments[attachment_index]
    name = _safe_filename(att.filename or f"attachment_{attachment_index}")
    dest = _inbox_dir() / f"{message.id}_{attachment_index}_{name}"
    await att.save(dest)
    kind = _guess_kind(dest, getattr(att, "content_type", None))
    return dest, kind


async def analyze_insurance_document_impl(
    *,
    text: Optional[str] = None,
    attachment_index: int = 0,
    file_url: Optional[str] = None,
    enable_vision: bool = True,
    record_id: Optional[str] = None,
    message=None,
) -> dict:
    """Core pipeline invocation used by the Chloride tool (and unit tests)."""
    path: Path | None = None
    kind = "text"
    source_text = (text or "").strip()

    try:
        if file_url:
            parsed = urlparse(file_url)
            name = _safe_filename(Path(parsed.path).name or "download.bin")
            msg_id = getattr(message, "id", "adhoc")
            path = _inbox_dir() / f"{msg_id}_url_{name}"
            await _download_url(file_url, path)
            kind = _guess_kind(path)
        elif not source_text and message is not None:
            path, kind = await _attachment_from_message(
                message, attachment_index=attachment_index
            )
        elif not source_text:
            return {
                "error": (
                    "Provide `text`, `file_url`, or attach a PDF/PNG to the Discord message."
                )
            }

        if path is not None and kind == "text":
            source_text = path.read_text(encoding="utf-8", errors="replace")
            path = None

        rid = record_id or f"discord-{getattr(message, 'id', 'adhoc')}"
        kwargs: dict = {
            "text": source_text,
            "record_id": rid,
            "enable_vision": enable_vision,
        }
        if path is not None:
            if kind == "pdf":
                kwargs["pdf_path"] = path
            elif kind == "image":
                kwargs["image_path"] = path
            else:
                return {
                    "error": (
                        f"Unsupported file type for pipeline: {path.suffix or kind}. "
                        "Use PDF, PNG/JPEG, or plain text."
                    ),
                    "path": str(path),
                }

        import asyncio

        result = await asyncio.to_thread(analyze_document, **kwargs)
        compact = compact_analysis(result)
        return {
            "ok": True,
            "analysis": compact,
            "discord_summary": format_discord_summary(compact),
            "source": {
                "kind": kind if path is not None else "text",
                "path": str(path) if path is not None else None,
                "repo_root": str(REPO_ROOT),
            },
        }
    except Exception as exc:
        logger.exception("analyze_insurance_document failed")
        return {"ok": False, "error": str(exc)}


def register_tools() -> None:
    """Attach pipeline tools onto the Chloride/Coral agent singleton."""
    global _TOOLS_REGISTERED
    if _TOOLS_REGISTERED:
        return

    try:
        from coral.agent import Deps, agent, restrict_tools_by_tier
        from pydantic_ai import RunContext
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "Chloride/Coral is not installed. Install with:\n"
            '  pip install -e ".[discord]"\n'
            "See discord/smol-doc-analyzer/README.md."
        ) from exc

    # Avoid double-registration if register_tools is called more than once.
    existing = getattr(agent, "_function_toolset", None)
    if existing is not None and "analyze_insurance_document" in getattr(existing, "tools", {}):
        _TOOLS_REGISTERED = True
        return

    async def analyze_insurance_document(
        ctx: RunContext[Deps],
        text: Optional[str] = None,
        attachment_index: int = 0,
        file_url: Optional[str] = None,
        enable_vision: bool = True,
        record_id: Optional[str] = None,
    ) -> dict:
        """
        Run the smol-doc-analyzer insurance document pipeline on text or a file.

        Chain: to_markdown → classify → extract → vision_llm → summarize.

        Prefer this tool for ACORD/loss-notice/claim documents instead of generic
        file analysis. Provide one of:
        - `text`: raw document text pasted by the user
        - `file_url`: Discord CDN URL, http(s) URL, or local path
        - otherwise uses `attachment_index` on the triggering Discord message

        Returns a compact analysis (document type, fields, memo, flags) plus a
        Discord-ready markdown summary string.
        """
        return await analyze_insurance_document_impl(
            text=text,
            attachment_index=attachment_index,
            file_url=file_url,
            enable_vision=enable_vision,
            record_id=record_id,
            message=getattr(ctx.deps, "message", None),
        )

    # Bind Deps into the nested function globals so pydantic-ai can resolve hints
    # under `from __future__ import annotations`.
    analyze_insurance_document.__globals__["Deps"] = Deps
    analyze_insurance_document.__globals__["RunContext"] = RunContext

    agent.tool(prepare=restrict_tools_by_tier)(analyze_insurance_document)
    _TOOLS_REGISTERED = True
    logger.info("Registered Chloride tool: analyze_insurance_document")


try:
    register_tools()
except ImportError:
    # Allow importing helpers without Chloride installed (unit tests / base install).
    pass
