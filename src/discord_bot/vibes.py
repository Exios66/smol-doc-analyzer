"""Guild DJ / vibe queue — voice playback when available, link queue otherwise."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Mood → search seeds for /vibe set
VIBE_SEEDS: dict[str, list[str]] = {
    "focus": [
        "lofi hip hop radio beats to relax study to",
        "ambient study music no vocals",
        "soft piano focus playlist",
    ],
    "chill": [
        "chill vibes playlist",
        "downtempo electronic chill",
        "sunset acoustic chill",
    ],
    "energy": [
        "upbeat indie dance playlist",
        "workout electronic energy",
        "feel good pop remix",
    ],
    "jazz": [
        "smooth jazz radio",
        "late night jazz trio",
        "coffee shop jazz playlist",
    ],
    "claims": [
        "calm instrumental office music",
        "soft electronic background music",
        "minimal deep house chill",
    ],
}


@dataclass
class Track:
    title: str
    url: str
    requested_by: str
    webpage_url: str | None = None
    duration: int | None = None

    def label(self) -> str:
        dur = f" ({self.duration // 60}:{self.duration % 60:02d})" if self.duration else ""
        return f"**{self.title}**{dur} · requested by {self.requested_by}"


@dataclass
class GuildVibeState:
    guild_id: int
    queue: list[Track] = field(default_factory=list)
    now_playing: Track | None = None
    mood: str = "chill"
    link_only: bool = False
    voice_channel_id: int | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "guild_id": self.guild_id,
            "mood": self.mood,
            "link_only": self.link_only,
            "now_playing": self.now_playing.title if self.now_playing else None,
            "queue_len": len(self.queue),
            "queue": [
                {"title": t.title, "url": t.webpage_url or t.url, "by": t.requested_by}
                for t in self.queue[:15]
            ],
        }


_STATES: dict[int, GuildVibeState] = {}
_LOCKS: dict[int, asyncio.Lock] = {}


def voice_deps_status() -> dict[str, bool]:
    ffmpeg = shutil.which("ffmpeg") is not None
    try:
        import nacl  # noqa: F401

        pynacl = True
    except ImportError:
        pynacl = False
    try:
        import yt_dlp  # noqa: F401

        ytdlp = True
    except ImportError:
        ytdlp = False
    return {"ffmpeg": ffmpeg, "pynacl": pynacl, "yt_dlp": ytdlp}


def can_play_voice() -> bool:
    s = voice_deps_status()
    return bool(s["ffmpeg"] and s["pynacl"] and s["yt_dlp"])


def get_state(guild_id: int) -> GuildVibeState:
    if guild_id not in _STATES:
        _STATES[guild_id] = GuildVibeState(guild_id=guild_id, link_only=not can_play_voice())
    return _STATES[guild_id]


def _lock(guild_id: int) -> asyncio.Lock:
    if guild_id not in _LOCKS:
        _LOCKS[guild_id] = asyncio.Lock()
    return _LOCKS[guild_id]


async def resolve_track(query: str, requested_by: str) -> Track:
    """Resolve a URL or search query to a playable Track via yt-dlp when available."""
    query = (query or "").strip()
    if not query:
        raise ValueError("Empty play query.")

    if not can_play_voice():
        # Link-only mode: treat query as a shareable URL / recommendation.
        title = query if len(query) < 80 else query[:77] + "…"
        return Track(title=title, url=query, webpage_url=query, requested_by=requested_by)

    import yt_dlp

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "default_search": "ytsearch1",
        "no_warnings": True,
    }

    def _extract() -> dict[str, Any]:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if info is None:
                raise RuntimeError("yt-dlp returned no info")
            if "entries" in info:
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    raise RuntimeError(f"No results for: {query}")
                info = entries[0]
            return info

    info = await asyncio.to_thread(_extract)
    return Track(
        title=str(info.get("title") or query),
        url=str(info.get("url") or info.get("webpage_url") or query),
        webpage_url=str(info.get("webpage_url") or query),
        duration=int(info["duration"]) if info.get("duration") else None,
        requested_by=requested_by,
    )


async def enqueue(guild_id: int, query: str, requested_by: str) -> Track:
    track = await resolve_track(query, requested_by)
    state = get_state(guild_id)
    async with _lock(guild_id):
        state.queue.append(track)
    return track


def format_queue(state: GuildVibeState) -> str:
    lines = ["## Vibes queue", f"Mood: **{state.mood}**"]
    if state.link_only or not can_play_voice():
        lines.append("_Link / recommendation mode_ (install `ffmpeg`, `PyNaCl`, `yt-dlp` for voice).")
    if state.now_playing:
        lines.append(f"Now: {state.now_playing.label()}")
    else:
        lines.append("Now: _(nothing)_")
    if not state.queue:
        lines.append("Up next: _(empty)_")
    else:
        lines.append("Up next:")
        for i, t in enumerate(state.queue[:12], 1):
            lines.append(f"{i}. {t.label()}")
    return "\n".join(lines)


async def ensure_voice(bot, interaction) -> tuple[Any | None, str | None]:
    """Join the caller's voice channel. Returns (voice_client, error)."""
    if not can_play_voice():
        return None, (
            "Voice DJ deps missing. Install optional extras: "
            '`pip install -e ".[discord,discord-voice]"` and ensure `ffmpeg` is on PATH. '
            "Falling back to link-queue mode — use `/play` to queue shareable links."
        )
    member = interaction.user
    voice_state = getattr(member, "voice", None)
    channel = getattr(voice_state, "channel", None) if voice_state else None
    if channel is None:
        return None, "Join a voice channel first, then run `/join` or `/play`."

    guild = interaction.guild
    vc = guild.voice_client if guild else None
    if vc and vc.is_connected():
        if vc.channel.id != channel.id:
            await vc.move_to(channel)
        return vc, None
    vc = await channel.connect()
    state = get_state(guild.id)
    state.voice_channel_id = channel.id
    state.link_only = False
    return vc, None


async def play_next(bot, guild_id: int) -> str | None:
    """Start the next queued track on the guild voice client. Returns status text."""
    state = get_state(guild_id)
    guild = bot.get_guild(guild_id)
    if guild is None:
        return "Guild not found."
    vc = guild.voice_client
    if vc is None or not vc.is_connected():
        return "Not connected to voice. Use `/join` first."

    async with _lock(guild_id):
        if not state.queue:
            state.now_playing = None
            return "Queue empty."
        track = state.queue.pop(0)
        state.now_playing = track

    if state.link_only or not can_play_voice():
        return f"Queued link (no voice): {track.label()}\n{track.webpage_url or track.url}"

    import discord

    def _after(err: Exception | None) -> None:
        if err:
            logger.warning("player error: %s", err)
        fut = asyncio.run_coroutine_threadsafe(play_next(bot, guild_id), bot.loop)
        try:
            fut.result(timeout=0.1)
        except Exception:
            pass

    source = discord.FFmpegPCMAudio(
        track.url,
        before_options="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
        options="-vn",
    )
    if vc.is_playing():
        vc.stop()
    vc.play(source, after=_after)
    return f"Now playing: {track.label()}"


async def stop_playback(guild_id: int, *, clear_queue: bool = False) -> str:
    state = get_state(guild_id)
    # Voice client stop is handled by caller who has bot reference
    async with _lock(guild_id):
        state.now_playing = None
        if clear_queue:
            state.queue.clear()
    return "Stopped." + (" Queue cleared." if clear_queue else "")


def set_mood(guild_id: int, mood: str) -> str:
    mood = (mood or "").strip().lower()
    if mood not in VIBE_SEEDS:
        known = ", ".join(sorted(VIBE_SEEDS))
        return f"Unknown mood `{mood}`. Try: {known}"
    state = get_state(guild_id)
    state.mood = mood
    seeds = VIBE_SEEDS[mood]
    return (
        f"Mood set to **{mood}**. Try `/play {seeds[0]}` "
        f"or ask me to keep the vibes going."
    )


def mood_seed(guild_id: int) -> str:
    state = get_state(guild_id)
    seeds = VIBE_SEEDS.get(state.mood) or VIBE_SEEDS["chill"]
    return seeds[0]
