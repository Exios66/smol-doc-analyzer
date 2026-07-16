"""Audio / voice-message transcription via OpenAI-compatible speech-to-text APIs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.utils.config import _load_dotenv, _secret

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".oga",
    ".webm",
    ".flac",
    ".mp4",
    ".mpeg",
    ".mpga",
}


def is_audio_path(path: Path, content_type: str | None = None) -> bool:
    if path.suffix.lower() in AUDIO_SUFFIXES:
        return True
    if content_type and (
        content_type.startswith("audio/")
        or content_type in {"video/webm", "application/ogg"}
    ):
        return True
    return False


def _stt_credentials() -> tuple[str, str, str]:
    """Return (api_key, base_url, model) for speech-to-text."""
    _load_dotenv()
    api_key = (
        _secret("DISCORD_AI_API_KEY")
        or _secret("OPENROUTER_API_KEY")
        or _secret("OPENAI_API_KEY")
        or _secret("AI_API_KEY")
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    # OpenRouter exposes OpenAI-compatible audio transcription for some models;
    # OpenAI Whisper remains the most reliable default when OPENAI_API_KEY is set.
    if _secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip():
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("DISCORD_STT_MODEL", "whisper-1")
        key = _secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip()
        return key, base, model

    if api_key:
        base = os.getenv(
            "DISCORD_STT_BASE_URL",
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ).rstrip("/")
        model = os.getenv("DISCORD_STT_MODEL", "openai/whisper-1")
        return api_key, base, model

    return "", "", ""


async def transcribe_audio_file(
    path: Path,
    *,
    language: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """
    Transcribe an audio file to text.

    Uses OpenAI Whisper when ``OPENAI_API_KEY`` is set; otherwise tries the
    OpenRouter OpenAI-compatible transcription endpoint with ``OPENROUTER_API_KEY``.
    """
    path = Path(path)
    if not path.exists():
        return {"ok": False, "error": f"Audio file not found: {path}"}
    if path.stat().st_size == 0:
        return {"ok": False, "error": "Audio file is empty."}

    api_key, base_url, model = _stt_credentials()
    if not api_key:
        return {
            "ok": False,
            "error": (
                "No speech-to-text credentials. Set OPENAI_API_KEY (preferred for Whisper) "
                "or OPENROUTER_API_KEY / DISCORD_AI_API_KEY in `.env`."
            ),
        }

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        return {"ok": False, "error": f"openai package required for transcription: {exc}"}

    import asyncio

    def _call() -> Any:
        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs: dict[str, Any] = {"model": model}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        with path.open("rb") as fh:
            return client.audio.transcriptions.create(file=fh, **kwargs)

    try:
        result = await asyncio.to_thread(_call)
    except Exception as exc:
        logger.exception("transcription failed for %s", path)
        return {"ok": False, "error": str(exc), "model": model, "base_url": base_url}

    text = getattr(result, "text", None) or str(result)
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "Transcription returned empty text.", "model": model}
    return {
        "ok": True,
        "text": text,
        "model": model,
        "path": str(path),
        "chars": len(text),
    }
