"""Discord slash (/) commands — docs pipeline + notes, STT, vibes, utilities.

Slash commands invoke capabilities directly (no LLM routing required). Chloride
chat (mention / `--` prefix) remains available for free-form agent behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import discord
from discord import app_commands

from src.discord_bot.formatters import format_discord_summary
from src.discord_bot.notes_store import format_notes, get_notes_store
from src.discord_bot.tools import (
    _guess_kind,
    _inbox_dir,
    _safe_filename,
    analyze_insurance_document_impl,
)
from src.discord_bot.transcribe import is_audio_path, transcribe_audio_file
from src.discord_bot.vibes import (
    VIBE_SEEDS,
    can_play_voice,
    enqueue,
    ensure_voice,
    format_queue,
    get_state,
    mood_seed,
    play_next,
    set_mood,
    stop_playback,
    voice_deps_status,
)
from src.utils.config import secrets_status

logger = logging.getLogger(__name__)

_MAX_CHUNK = 1900

_SLASH_COMMANDS = (
    ("analyze", "Analyze an insurance document from text or an attached PDF/PNG"),
    ("analyze_url", "Analyze an insurance document downloaded from a URL"),
    ("note", "Capture, search, and list server notes / transcripts"),
    ("transcribe", "Transcribe a voice message or audio attachment"),
    ("play", "Queue a track or link for the DJ / vibes system"),
    ("queue", "Show the vibes queue"),
    ("skip", "Skip the current track"),
    ("stop", "Stop playback and optionally clear the queue"),
    ("join", "Join your voice channel"),
    ("leave", "Leave the voice channel"),
    ("vibe", "Set or show the server mood"),
    ("poll", "Create a quick server poll"),
    ("remind", "Save a reminder note for the team"),
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


def _guild_id(interaction: discord.Interaction) -> str:
    return str(interaction.guild_id or "dm")


def _format_status_message() -> str:
    from src.utils.config import _load_dotenv, _secret

    _load_dotenv()
    status = secrets_status()
    discord_set = bool(_secret("DISCORD_TOKEN"))
    voice = voice_deps_status()
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
        "### Voice DJ deps",
        f"- ffmpeg: `{'yes' if voice['ffmpeg'] else 'no'}`",
        f"- PyNaCl: `{'yes' if voice['pynacl'] else 'no'}`",
        f"- yt-dlp: `{'yes' if voice['yt_dlp'] else 'no'}`",
        f"- Voice playback ready: `{'yes' if can_play_voice() else 'no (link-queue mode)'}`",
        "",
        "### Roles",
        "- Docs: `/analyze` · Notes: `/note` · STT: `/transcribe` · DJ: `/play`",
        "- Chat: mention the bot or prefix with `--`",
        "- Pipeline: `to_markdown → classify → extract → vision_llm → summarize`",
    ]
    return "\n".join(lines)


def _help_message() -> str:
    lines = [
        "## Slash commands",
        "",
        "All-purpose server agent — docs specialty plus notes, transcription, and vibes:",
        "",
    ]
    for name, desc in _SLASH_COMMANDS:
        lines.append(f"- `/{name}` — {desc}")
    lines.extend(
        [
            "",
            "### Tips",
            "- `/analyze` accepts pasted `text` and/or a PDF/PNG `attachment`",
            "- `/transcribe` turns voice notes into text (and saves a transcript note)",
            "- `/play` queues music; without ffmpeg/PyNaCl/yt-dlp it queues shareable links",
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
    local_path = None

    if attachment is not None:
        name = _safe_filename(getattr(attachment, "filename", None) or "attachment.bin")
        dest = _inbox_dir() / f"slash_{interaction.id}_{name}"
        await attachment.save(dest)
        kind = _guess_kind(dest, getattr(attachment, "content_type", None))
        if kind == "text":
            source_text = dest.read_text(encoding="utf-8", errors="replace")
        else:
            # Trusted inbox file — pass as local_path, never via file_url download.
            local_path = dest

    out = await analyze_insurance_document_impl(
        text=source_text,
        file_url=resolved_url if local_path is None else None,
        local_path=local_path,
        enable_vision=vision,
        record_id=f"discord-slash-{interaction.id}",
        message=None,
    )

    if not out.get("ok"):
        err = out.get("error") or "Analysis failed."
        return f"## Analysis failed\n{err}"
    return out.get("discord_summary") or format_discord_summary(out.get("analysis") or {})


def register_slash_commands(bot: Any) -> None:
    """Register all slash commands / groups on ``bot.tree``."""
    tree = getattr(bot, "tree", None)
    if tree is None:
        raise AttributeError("Bot has no CommandTree; expected Chloride CoralBot.tree")

    existing = {cmd.name for cmd in tree.get_commands()}
    if "analyze" in existing and "note" in existing and "help" in existing:
        logger.info("Slash commands already registered on bot.tree")
        return

    # ---- Document pipeline -------------------------------------------------
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

    # ---- Notes -------------------------------------------------------------
    note = app_commands.Group(name="note", description="Server notes, transcripts, and reminders")

    @note.command(name="add", description="Save a note for this server")
    @app_commands.describe(
        body="Note text",
        title="Optional short title",
        tags="Comma-separated tags",
    )
    async def note_add(
        interaction: discord.Interaction,
        body: str,
        title: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        store = get_notes_store()
        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
        try:
            n = store.add(
                guild_id=_guild_id(interaction),
                channel_id=str(interaction.channel_id or "dm"),
                author_id=str(interaction.user.id),
                author_name=interaction.user.display_name,
                body=body,
                title=title or "",
                tags=tag_list,
                source="slash",
            )
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_message(
            f"Saved note `{n.note_id}` — **{n.title}**",
            ephemeral=False,
        )

    @note.command(name="list", description="List recent notes")
    @app_commands.describe(limit="How many notes to show (max 20)", kind="Filter: note|transcript|reminder")
    async def note_list(
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 20] = 8,
        kind: Optional[str] = None,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        notes = get_notes_store().list_recent(_guild_id(interaction), kind=kind, limit=limit)
        await interaction.response.send_message(format_notes(notes, heading="Recent notes"))

    @note.command(name="search", description="Search notes by keyword")
    @app_commands.describe(query="Search text", limit="Max results")
    async def note_search(
        interaction: discord.Interaction,
        query: str,
        limit: app_commands.Range[int, 1, 20] = 8,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        notes = get_notes_store().search(_guild_id(interaction), query, limit=limit)
        await interaction.response.send_message(
            format_notes(notes, heading=f"Notes matching `{query}`")
        )

    @note.command(name="delete", description="Delete a note by id")
    @app_commands.describe(note_id="Note id from /note list")
    async def note_delete(interaction: discord.Interaction, note_id: str) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        ok = get_notes_store().delete(note_id.strip(), guild_id=_guild_id(interaction))
        await interaction.response.send_message(
            f"Deleted `{note_id}`." if ok else f"No note `{note_id}` in this server.",
            ephemeral=True,
        )

    @note.command(name="show", description="Show a full note by id")
    async def note_show(interaction: discord.Interaction, note_id: str) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        n = get_notes_store().get(note_id.strip())
        if n is None or n.guild_id != _guild_id(interaction):
            await interaction.response.send_message("Note not found.", ephemeral=True)
            return
        body = f"## {n.title}\n`{n.note_id}` · {n.kind} · {n.author_name}\n\n{n.body}"
        await interaction.response.send_message(body[:1900])

    tree.add_command(note)

    # ---- Transcription -----------------------------------------------------
    @tree.command(name="transcribe", description="Transcribe a voice note or audio attachment")
    @app_commands.describe(
        attachment="Audio file (ogg/mp3/wav/m4a/webm)",
        save_note="Save transcript to /note log (default true)",
        language="Optional BCP-47 language hint (e.g. en)",
    )
    async def transcribe_cmd(
        interaction: discord.Interaction,
        attachment: discord.Attachment,
        save_note: bool = True,
        language: Optional[str] = None,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        await interaction.response.defer(thinking=True)
        name = _safe_filename(attachment.filename or "audio.bin")
        dest = _inbox_dir() / f"stt_slash_{interaction.id}_{name}"
        await attachment.save(dest)
        if not is_audio_path(dest, attachment.content_type):
            await _send_chunks(
                interaction,
                f"Attachment `{name}` does not look like audio. "
                "Supported: mp3, wav, m4a, ogg, webm, flac.",
                ephemeral=True,
            )
            return
        result = await transcribe_audio_file(dest, language=language)
        if not result.get("ok"):
            await _send_chunks(interaction, f"## Transcription failed\n{result.get('error')}")
            return
        text = result["text"]
        extra = ""
        if save_note:
            n = get_notes_store().add(
                guild_id=_guild_id(interaction),
                channel_id=str(interaction.channel_id or "dm"),
                author_id=str(interaction.user.id),
                author_name=interaction.user.display_name,
                body=text,
                title=f"Transcript · {name}",
                kind="transcript",
                tags=["transcript"],
                source=str(dest),
            )
            extra = f"\n\n_Saved as note `{n.note_id}`_"
        await _send_chunks(interaction, f"## Transcript\n\n{text}{extra}")

    # ---- Vibes / DJ --------------------------------------------------------
    @tree.command(name="join", description="Join your current voice channel")
    async def join_cmd(interaction: discord.Interaction) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        if interaction.guild is None:
            await interaction.response.send_message("Voice only works in a server.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        vc, err = await ensure_voice(bot, interaction)
        if err and vc is None and not can_play_voice():
            await interaction.followup.send(err, ephemeral=True)
            return
        if err and vc is None:
            await interaction.followup.send(err, ephemeral=True)
            return
        await interaction.followup.send(
            f"Joined **{vc.channel.name}**." if vc else (err or "Joined."),
            ephemeral=True,
        )

    @tree.command(name="leave", description="Leave the voice channel")
    async def leave_cmd(interaction: discord.Interaction) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        guild = interaction.guild
        if guild is None or guild.voice_client is None:
            await interaction.response.send_message("Not in a voice channel.", ephemeral=True)
            return
        await guild.voice_client.disconnect()
        await stop_playback(guild.id, clear_queue=False)
        await interaction.response.send_message("Left voice.", ephemeral=True)

    @tree.command(name="play", description="Queue a track (YouTube URL / search) or shareable link")
    @app_commands.describe(query="YouTube URL, search text, or any music link")
    async def play_cmd(interaction: discord.Interaction, query: str) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        if interaction.guild is None:
            await interaction.response.send_message("DJ only works in a server.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        guild_id = interaction.guild.id
        try:
            track = await enqueue(guild_id, query, interaction.user.display_name)
        except Exception as exc:  # noqa: BLE001
            await _send_chunks(interaction, f"Could not queue: `{exc}`")
            return

        status_bits = [f"Queued: {track.label()}"]
        if track.webpage_url:
            status_bits.append(track.webpage_url)

        if can_play_voice():
            vc, err = await ensure_voice(bot, interaction)
            if vc is not None:
                state = get_state(guild_id)
                if state.now_playing is None and not (vc.is_playing() if vc else False):
                    msg = await play_next(bot, guild_id)
                    if msg:
                        status_bits.append(msg)
            elif err:
                status_bits.append(f"_(voice)_ {err}")
        else:
            status_bits.append("_Link-queue mode_ — click the URL above.")

        status_bits.append("")
        status_bits.append(format_queue(get_state(guild_id)))
        await _send_chunks(interaction, "\n".join(status_bits))

    @tree.command(name="queue", description="Show the vibes / DJ queue")
    async def queue_cmd(interaction: discord.Interaction) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        if interaction.guild is None:
            await interaction.response.send_message("No guild queue in DMs.", ephemeral=True)
            return
        await interaction.response.send_message(format_queue(get_state(interaction.guild.id)))

    @tree.command(name="skip", description="Skip the current track")
    async def skip_cmd(interaction: discord.Interaction) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message("No guild.", ephemeral=True)
            return
        await interaction.response.defer()
        vc = guild.voice_client
        if vc and vc.is_playing():
            vc.stop()  # after-callback advances queue
            await interaction.followup.send("Skipped.")
        else:
            msg = await play_next(bot, guild.id)
            await interaction.followup.send(msg or "Nothing to skip.")

    @tree.command(name="stop", description="Stop playback and clear the queue")
    async def stop_cmd(interaction: discord.Interaction) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message("No guild.", ephemeral=True)
            return
        vc = guild.voice_client
        if vc and (vc.is_playing() or vc.is_paused()):
            vc.stop()
        msg = await stop_playback(guild.id, clear_queue=True)
        await interaction.response.send_message(msg)

    @tree.command(name="vibe", description="Set or show the server mood for DJ suggestions")
    @app_commands.describe(mood="focus | chill | energy | jazz | claims (omit to show status)")
    async def vibe_cmd(interaction: discord.Interaction, mood: Optional[str] = None) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        if interaction.guild is None:
            await interaction.response.send_message("Vibes are per-server.", ephemeral=True)
            return
        if mood:
            msg = set_mood(interaction.guild.id, mood)
            await interaction.response.send_message(msg)
            return
        state = get_state(interaction.guild.id)
        seeds = ", ".join(f"`{s}`" for s in VIBE_SEEDS.get(state.mood, [])[:2])
        await interaction.response.send_message(
            f"Current mood: **{state.mood}**\nSuggestions: {seeds}\n"
            f"Try `/play {mood_seed(interaction.guild.id)}`"
        )

    # ---- Utility -----------------------------------------------------------
    @tree.command(name="poll", description="Create a quick yes/no or multi-option poll")
    @app_commands.describe(
        question="Poll question",
        options="Optional comma-separated choices (default: Yes / No)",
    )
    async def poll_cmd(
        interaction: discord.Interaction,
        question: str,
        options: Optional[str] = None,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        choices = [c.strip() for c in (options or "Yes,No").split(",") if c.strip()]
        if len(choices) < 2:
            await interaction.response.send_message("Need at least 2 options.", ephemeral=True)
            return
        if len(choices) > 10:
            await interaction.response.send_message("Max 10 options.", ephemeral=True)
            return
        # Number emoji reactions 1️⃣..
        numerals = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        lines = [f"**Poll:** {question}", ""]
        for i, choice in enumerate(choices):
            lines.append(f"{numerals[i]} {choice}")
        await interaction.response.send_message("\n".join(lines))
        msg = await interaction.original_response()
        for i in range(len(choices)):
            await msg.add_reaction(numerals[i])

    @tree.command(name="remind", description="Save a reminder note for the team")
    @app_commands.describe(text="What to remember", title="Optional title")
    async def remind_cmd(
        interaction: discord.Interaction,
        text: str,
        title: Optional[str] = None,
    ) -> None:
        if not await _ensure_allowed(interaction, bot):
            return
        n = get_notes_store().add(
            guild_id=_guild_id(interaction),
            channel_id=str(interaction.channel_id or "dm"),
            author_id=str(interaction.user.id),
            author_name=interaction.user.display_name,
            body=text,
            title=title or "Reminder",
            kind="reminder",
            tags=["reminder"],
            source="slash",
        )
        await interaction.response.send_message(
            f"Reminder saved as `{n.note_id}` — find it later with `/note search reminder`."
        )

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
