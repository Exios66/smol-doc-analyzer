"""Additional Chloride agent tools: notes, transcription, vibes, utility."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from src.discord_bot.notes_store import format_notes, get_notes_store
from src.discord_bot.transcribe import is_audio_path, transcribe_audio_file
from src.discord_bot.vibes import (
    enqueue,
    format_queue,
    get_state,
    mood_seed,
    set_mood,
    voice_deps_status,
)

if TYPE_CHECKING:
    from coral.agent import Deps
    from pydantic_ai import RunContext

logger = logging.getLogger(__name__)
_EXTRAS_REGISTERED = False


def _guild_id_from_ctx(ctx) -> str:
    message = getattr(getattr(ctx, "deps", None), "message", None)
    guild = getattr(message, "guild", None) if message else None
    if guild is not None:
        return str(guild.id)
    channel = getattr(message, "channel", None) if message else None
    guild2 = getattr(channel, "guild", None) if channel else None
    if guild2 is not None:
        return str(guild2.id)
    return "dm"


def _author_from_ctx(ctx) -> tuple[str, str, str]:
    message = getattr(getattr(ctx, "deps", None), "message", None)
    author = getattr(message, "author", None) if message else None
    channel = getattr(message, "channel", None) if message else None
    author_id = str(getattr(author, "id", "0"))
    author_name = getattr(author, "display_name", None) or getattr(author, "name", "user")
    channel_id = str(getattr(channel, "id", "dm"))
    return author_id, author_name, channel_id


async def _save_attachment_audio(message, attachment_index: int = 0):
    from src.discord_bot.tools import _attachment_from_message, _guess_kind

    path, kind = await _attachment_from_message(message, attachment_index=attachment_index)
    # kind may be unknown for audio; check path/content-type
    att = message.attachments[attachment_index]
    if not is_audio_path(path, getattr(att, "content_type", None)) and kind not in {
        "unknown",
        "text",
    }:
        # still allow if suffix looks audio
        if not is_audio_path(path):
            raise ValueError(f"Attachment does not look like audio: {path.name}")
    return path


def register_extra_tools() -> None:
    """Register notes / STT / vibes / utility tools on the Chloride agent."""
    global _EXTRAS_REGISTERED
    if _EXTRAS_REGISTERED:
        return

    try:
        from coral.agent import Deps, agent, restrict_tools_by_tier
        from pydantic_ai import RunContext
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Chloride/Coral is not installed. Install with:\n"
            '  pip install -e ".[discord]"'
        ) from exc

    existing = getattr(agent, "_function_toolset", None)
    tools = getattr(existing, "tools", {}) if existing is not None else {}
    if "save_note" in tools:
        _EXTRAS_REGISTERED = True
        return

    async def save_note(
        ctx: RunContext[Deps],
        body: str,
        title: str = "",
        tags: Optional[str] = None,
        kind: str = "note",
    ) -> dict:
        """
        Save a server note / transcript / reminder for later retrieval.

        Use for meeting notes, claim follow-ups, decisions, or anything the
        team should remember. `tags` is a comma-separated string.
        `kind` is one of: note, transcript, reminder.
        """
        store = get_notes_store()
        author_id, author_name, channel_id = _author_from_ctx(ctx)
        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
        note = store.add(
            guild_id=_guild_id_from_ctx(ctx),
            channel_id=channel_id,
            author_id=author_id,
            author_name=author_name,
            body=body,
            title=title,
            kind=kind or "note",
            tags=tag_list,
            source="agent",
        )
        return {"ok": True, "note": note.to_dict()}

    async def search_notes(
        ctx: RunContext[Deps],
        query: str = "",
        limit: int = 8,
        kind: Optional[str] = None,
    ) -> dict:
        """
        Search saved guild notes by keyword (title/body/tags/author).
        Empty query returns the most recent notes. Optional `kind` filter:
        note | transcript | reminder.
        """
        store = get_notes_store()
        guild_id = _guild_id_from_ctx(ctx)
        if query.strip():
            notes = store.search(guild_id, query, limit=limit)
        else:
            notes = store.list_recent(guild_id, kind=kind, limit=limit)
        if kind and query.strip():
            notes = [n for n in notes if n.kind == kind]
        return {
            "ok": True,
            "count": len(notes),
            "notes": [n.to_dict() for n in notes],
            "discord_summary": format_notes(notes, heading="Notes search"),
        }

    async def transcribe_audio(
        ctx: RunContext[Deps],
        attachment_index: int = 0,
        file_url: Optional[str] = None,
        save_as_note: bool = True,
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe a Discord voice message or audio attachment to text.

        Prefer this when a user uploads .ogg/.mp3/.wav/.m4a/.webm or asks to
        turn spoken notes into text. Set save_as_note=True (default) to store
        the transcript in the guild note log.
        """
        from src.discord_bot.tools import _download_url, _inbox_dir, _safe_filename
        from urllib.parse import urlparse
        from pathlib import Path

        message = getattr(ctx.deps, "message", None)
        path = None
        try:
            if file_url:
                parsed = urlparse(file_url)
                name = _safe_filename(Path(parsed.path).name or "audio.bin")
                dest = _inbox_dir() / f"stt_{getattr(message, 'id', 'adhoc')}_{name}"
                path = await _download_url(file_url, dest)
            elif message is not None:
                path = await _save_attachment_audio(message, attachment_index=attachment_index)
            else:
                return {
                    "ok": False,
                    "error": "Provide an audio attachment or file_url to transcribe.",
                }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        result = await transcribe_audio_file(path, language=language)
        if not result.get("ok"):
            return result

        note_info = None
        if save_as_note:
            store = get_notes_store()
            author_id, author_name, channel_id = _author_from_ctx(ctx)
            note = store.add(
                guild_id=_guild_id_from_ctx(ctx),
                channel_id=channel_id,
                author_id=author_id,
                author_name=author_name,
                body=result["text"],
                title="Transcript",
                kind="transcript",
                tags=["transcript"],
                source=str(path),
            )
            note_info = note.to_dict()

        return {
            "ok": True,
            "text": result["text"],
            "model": result.get("model"),
            "note": note_info,
            "discord_summary": f"## Transcript\n\n{result['text']}",
        }

    async def vibe_control(
        ctx: RunContext[Deps],
        action: str = "status",
        query: Optional[str] = None,
        mood: Optional[str] = None,
    ) -> dict:
        """
        Control the server DJ / vibes queue.

        Actions:
        - status: show now-playing + queue
        - play: enqueue `query` (YouTube URL or search text) and report it
        - mood: set vibe mood (focus, chill, energy, jazz, claims)
        - seed: suggest a track query for the current mood

        Voice playback needs ffmpeg + PyNaCl + yt-dlp; otherwise operates as a
        link recommendation queue the server can click.
        """
        message = getattr(ctx.deps, "message", None)
        guild = getattr(message, "guild", None) if message else None
        if guild is None:
            return {"ok": False, "error": "Vibes only work inside a guild (not DMs)."}
        guild_id = int(guild.id)
        author = getattr(message, "author", None)
        by = getattr(author, "display_name", None) or getattr(author, "name", "user")
        action = (action or "status").strip().lower()

        if action == "mood":
            if not mood:
                return {"ok": False, "error": "Provide mood=focus|chill|energy|jazz|claims"}
            msg = set_mood(guild_id, mood)
            return {"ok": True, "discord_summary": msg, "state": get_state(guild_id).snapshot()}

        if action == "seed":
            seed = mood_seed(guild_id)
            return {
                "ok": True,
                "query": seed,
                "discord_summary": f"Suggested play query for current mood: `{seed}`",
            }

        if action == "play":
            q = (query or "").strip() or mood_seed(guild_id)
            track = await enqueue(guild_id, q, by)
            state = get_state(guild_id)
            return {
                "ok": True,
                "track": {
                    "title": track.title,
                    "url": track.webpage_url or track.url,
                },
                "voice_deps": voice_deps_status(),
                "discord_summary": (
                    f"Queued: {track.label()}\n{track.webpage_url or track.url}\n\n"
                    f"{format_queue(state)}"
                ),
                "state": state.snapshot(),
            }

        # status default
        state = get_state(guild_id)
        return {
            "ok": True,
            "voice_deps": voice_deps_status(),
            "discord_summary": format_queue(state),
            "state": state.snapshot(),
        }

    async def server_help(ctx: RunContext[Deps]) -> dict:
        """
        Summarize what this Discord agent can do in this server
        (docs, notes, transcription, vibes, chat, search).
        """
        text = (
            "## Server agent capabilities\n\n"
            "- **Insurance docs**: `analyze_insurance_document` or `/analyze`\n"
            "- **Notes**: `save_note` / `search_notes` or `/note`\n"
            "- **Transcription**: `transcribe_audio` or `/transcribe`\n"
            "- **DJ / vibes**: `vibe_control` or `/play` `/queue` `/vibe`\n"
            "- **Chat**: mention me or prefix with `--`\n"
            "- **Search**: `duckduckgo_search`, `search_discord`\n"
            "- **Files**: `analyse_file` for non-insurance media\n"
        )
        return {"ok": True, "discord_summary": text}

    for fn in (
        save_note,
        search_notes,
        transcribe_audio,
        vibe_control,
        server_help,
    ):
        fn.__globals__["Deps"] = Deps
        fn.__globals__["RunContext"] = RunContext
        agent.tool(prepare=restrict_tools_by_tier)(fn)

    _EXTRAS_REGISTERED = True
    logger.info(
        "Registered Chloride tools: save_note, search_notes, transcribe_audio, "
        "vibe_control, server_help"
    )


try:
    register_extra_tools()
except ImportError:
    pass
