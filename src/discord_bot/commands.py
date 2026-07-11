"""Discord slash (/) commands that invoke the document-analysis pipeline directly.

These are actionable app commands users can run from Discord's command picker
without relying on the LLM agent to choose a tool. Chloride's CoralBot already
owns a ``CommandTree`` and syncs it on ready; we register additional commands
onto that tree before ``client.run``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.discord_bot.formatters import format_discord_summary
from src.discord_bot.tools import (
    _guess_kind,
    _inbox_dir,
    _safe_filename,
    analyze_insurance_document_impl,
)
from src.utils.config import secrets_status

logger = logging.getLogger(__name__)

# Discord message content hard limit is 2000; leave headroom for formatting.
_MAX_CHUNK = 1900

_SLASH_COMMANDS = (
    ("analyze", "Analyze an insurance document from text or an attached PDF/PNG"),
    ("analyze_url", "Analyze an insurance document downloaded from a URL"),
    ("status", "Show bot and secret readiness (never prints secret values)"),
    ("help", "List actionable slash commands for this bot"),
    ("ping", "Check that the bot is online and measure gateway latency"),
)


def chunk_message(text: str, limit: int = _MAX_CHUNK) -> list[str]:
    """Split long Discord replies into <=limit character chunks on line boundaries."""
    text = (text or "").strip() or "(empty)"
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                chunks.append("".join(current).rstrip())
                current, size = [], 0
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
            continue
        if size + len(line) > limit and current:
            chunks.append("".join(current).rstrip())
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current).rstrip())
    return chunks or [text[:limit]]


async def _send_chunks(interaction: Any, text: str, *, ephemeral: bool = False) -> None:
    parts = chunk_message(text)
    first = True
    for part in parts:
        if first:
            await interaction.followup.send(part, ephemeral=ephemeral)
            first = False
        else:
            await interaction.followup.send(part, ephemeral=ephemeral)


async def _ensure_allowed(interaction: Any, bot: Any) -> bool:
    """Respect Chloride tier / allow-list when available."""
    may_chat = getattr(bot, "_may_chat", None)
    if may_chat is None:
        return True
    allowed, _tier = may_chat(interaction.user)
    if allowed:
        return True
    msg = "You are not allowed to use this bot in this server."
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)
    return False


def _format_status_message() -> str:
    from src.utils.config import _load_dotenv, _secret

    _load_dotenv()
    status = secrets_status()
    discord_set = bool(_secret("DISCORD_TOKEN"))
    lines = [
        "## Bot status",
        "",
        f"- `.env` file present: `{'yes' if status.get('dotenv_file') else 'no'}`",
        f"- `DISCORD_TOKEN` set: `{'yes' if discord_set else 'no'}`",
        f"- `OPENROUTER_API_KEY` set: `{'yes' if status.get('OPENROUTER_API_KEY') else 'no'}`",
        f"- `WANDB_API_KEY` set: `{'yes' if status.get('WANDB_API_KEY') else 'no'}`",
        f"- `HF_TOKEN` set: `{'yes' if status.get('HF_TOKEN') else 'no'}`",
        f"- Vision model path set: `{'yes' if status.get('VISION_LLM_MODEL_PATH') else 'no'}`",
        "",
        "Pipeline chain: `to_markdown → classify → extract → vision_llm → summarize`",
        "Use `/analyze` or `/analyze_url` to run it from Discord.",
    ]
    return "\n".join(lines)


def _help_message() -> str:
    lines = [
        "## Slash commands",
        "",
        "These run the local document pipeline **directly** (no LLM tool routing required):",
        "",
    ]
    for name, desc in _SLASH_COMMANDS:
        lines.append(f"- `/{name}` — {desc}")
    lines.extend(
        [
            "",
            "### Tips",
            "- `/analyze` accepts pasted `text` and/or a PDF/PNG `attachment`",
            "- Mention the bot or use the `--` prefix for free-form Chloride chat",
            "- Right-click a message → **Apps → Ask Me** to analyze that message",
        ]
    )
    return "\n".join(lines)


async def _run_analyze_from_interaction(
    interaction: Any,
    *,
    text: Optional[str] = None,
    attachment: Any | None = None,
    file_url: Optional[str] = None,
    vision: bool = True,
) -> str:
    """Execute analysis and return a Discord markdown summary (or error string)."""
    source_text = (text or "").strip() or None
    resolved_url = (file_url or "").strip() or None

    if attachment is not None:
        name = _safe_filename(getattr(attachment, "filename", None) or "attachment.bin")
        dest = _inbox_dir() / f"slash_{interaction.id}_{name}"
        await attachment.save(dest)
        kind = _guess_kind(dest, getattr(attachment, "content_type", None))
        if kind == "text":
            source_text = dest.read_text(encoding="utf-8", errors="replace")
        else:
            resolved_url = str(dest)

    out = await analyze_insurance_document_impl(
        text=source_text,
        file_url=resolved_url,
        enable_vision=vision,
        record_id=f"discord-slash-{interaction.id}",
        message=None,
    )

    if not out.get("ok"):
        err = out.get("error") or "Analysis failed."
        return f"## Analysis failed\n{err}"
    return out.get("discord_summary") or format_discord_summary(out.get("analysis") or {})


def register_slash_commands(bot: Any) -> None:
    """Register `/analyze`, `/analyze_url`, `/status`, `/help`, `/ping` on ``bot.tree``."""
    import discord
    from discord import app_commands

    tree = getattr(bot, "tree", None)
    if tree is None:
        raise AttributeError("Bot has no CommandTree; expected Chloride CoralBot.tree")

    # Avoid duplicate registration if runner is invoked twice in one process.
    existing = {cmd.name for cmd in tree.get_commands()}
    if "analyze" in existing and "help" in existing:
        logger.info("Slash commands already registered on bot.tree")
        return

    @tree.command(
        name="analyze",
        description="Analyze an insurance document from text or an attached PDF/PNG",
    )
    @app_commands.describe(
        text="Paste document text (loss notice, ACORD form text, etc.)",
        attachment="Optional PDF or image attachment to analyze",
        vision="Enable the vision_llm refine stage (default: true)",
    )
    async def analyze_cmd(
        interaction: discord.Interaction,
        text: Optional[str] = None,
        attachment: Optional[discord.Attachment] = None,
        vision: bool = True,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        if not (text and text.strip()) and attachment is None:
            await interaction.response.send_message(
                "Provide `text` and/or an `attachment` (PDF/PNG).",
                ephemeral=True,
            )
            return
        await interaction.response.defer(thinking=True)
        try:
            summary = await _run_analyze_from_interaction(
                interaction,
                text=text,
                attachment=attachment,
                vision=vision,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("/analyze failed")
            summary = f"## Analysis failed\n`{exc}`"
        await _send_chunks(interaction, summary)

    @tree.command(
        name="analyze_url",
        description="Analyze an insurance document downloaded from a URL",
    )
    @app_commands.describe(
        url="HTTP(S) URL or local path to a PDF/PNG/text document",
        vision="Enable the vision_llm refine stage (default: true)",
    )
    async def analyze_url_cmd(
        interaction: discord.Interaction,
        url: str,
        vision: bool = True,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        if not (url or "").strip():
            await interaction.response.send_message("Provide a non-empty `url`.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            summary = await _run_analyze_from_interaction(
                interaction,
                file_url=url.strip(),
                vision=vision,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("/analyze_url failed")
            summary = f"## Analysis failed\n`{exc}`"
        await _send_chunks(interaction, summary)

    @tree.command(
        name="status",
        description="Show bot and secret readiness (never prints secret values)",
    )
    async def status_cmd(interaction: discord.Interaction) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        await interaction.response.send_message(_format_status_message(), ephemeral=True)

    @tree.command(name="help", description="List actionable slash commands for this bot")
    async def help_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(_help_message(), ephemeral=True)

    @tree.command(name="ping", description="Check that the bot is online")
    async def ping_cmd(interaction: discord.Interaction) -> None:
        latency_ms = getattr(bot, "latency", None)
        if isinstance(latency_ms, (int, float)):
            body = f"Pong · gateway latency **{latency_ms * 1000:.0f} ms**"
        else:
            body = "Pong"
        await interaction.response.send_message(body, ephemeral=True)

    logger.info(
        "Registered Discord slash commands: %s",
        ", ".join(name for name, _ in _SLASH_COMMANDS),
    )
