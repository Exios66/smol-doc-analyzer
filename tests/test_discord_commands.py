"""Tests for Discord slash-command helpers and new agent modules."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.discord_bot.commands import (
    _SLASH_COMMANDS,
    _format_status_message,
    _help_message,
    _run_analyze_from_interaction,
    chunk_message,
    register_slash_commands,
)
from src.discord_bot.notes_store import NotesStore, format_notes
from src.discord_bot.vibes import VIBE_SEEDS, format_queue, get_state, set_mood


def test_chunk_message_splits_long_text():
    text = "\n".join(f"line-{i}-" + ("x" * 80) for i in range(40))
    parts = chunk_message(text, limit=500)
    assert len(parts) > 1
    assert all(len(p) <= 500 for p in parts)
    rejoined = "\n".join(parts)
    assert "line-0" in rejoined and "line-39" in rejoined


def test_chunk_message_short():
    assert chunk_message("hello") == ["hello"]


def test_help_message_lists_commands():
    text = _help_message()
    for name, _desc in _SLASH_COMMANDS:
        assert f"/{name}" in text
    assert "/note" in text
    assert "/transcribe" in text
    assert "/play" in text


def test_status_message_has_no_secret_values(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-should-not-appear")
    monkeypatch.setenv("DISCORD_TOKEN", "discord-secret-should-not-appear")
    text = _format_status_message()
    assert "sk-secret-should-not-appear" not in text
    assert "discord-secret-should-not-appear" not in text
    assert "DISCORD_TOKEN" in text
    assert "yes" in text
    assert "Voice DJ deps" in text


def test_register_slash_commands_on_mock_tree():
    discord = pytest.importorskip("discord")
    if not hasattr(discord, "app_commands"):
        pytest.skip("discord.py app_commands unavailable in this environment")

    class FakeTree:
        def __init__(self):
            self._commands = {}

        def command(self, *, name, description):
            def deco(fn):
                self._commands[name] = fn
                fn.name = name
                return fn

            return deco

        def add_command(self, group):
            self._commands[group.name] = group

        def get_commands(self):
            return list(self._commands.values())

    bot = SimpleNamespace(tree=FakeTree(), latency=0.042, _may_chat=lambda user: (True, None))
    register_slash_commands(bot)
    names = {getattr(c, "name", None) for c in bot.tree.get_commands()}
    assert names >= {
        "analyze",
        "analyze_url",
        "note",
        "transcribe",
        "play",
        "queue",
        "vibe",
        "poll",
        "remind",
        "status",
        "help",
        "ping",
    }
    # Idempotent
    before = len(bot.tree.get_commands())
    register_slash_commands(bot)
    assert len(bot.tree.get_commands()) == before


def test_run_analyze_from_interaction_text():
    interaction = SimpleNamespace(id=4242)
    out = asyncio.run(
        _run_analyze_from_interaction(
            interaction,
            text=(
                "AUTOMOBILE LOSS NOTICE\n"
                "Claim Number: CLM-SLASH-1\n"
                "Date of Loss: 2024-01-15\n"
                "Loss Type: collision\n"
            ),
            vision=False,
        )
    )
    assert "## Document analysis" in out
    assert "Type:" in out


def test_notes_store_roundtrip(tmp_path: Path):
    store = NotesStore(db_path=tmp_path / "notes.db")
    n = store.add(
        guild_id="g1",
        channel_id="c1",
        author_id="u1",
        author_name="Ada",
        body="Follow up on claim CLM-1 tomorrow",
        title="Follow-up",
        tags=["claims", "todo"],
    )
    assert n.note_id
    found = store.search("g1", "CLM-1")
    assert len(found) == 1
    assert found[0].title == "Follow-up"
    recent = store.list_recent("g1", limit=5)
    assert recent[0].note_id == n.note_id
    assert "Follow-up" in format_notes(recent)
    assert store.delete(n.note_id, guild_id="g1")
    assert store.search("g1", "CLM-1") == []


def test_vibes_mood_and_queue_format():
    gid = 999001
    msg = set_mood(gid, "focus")
    assert "focus" in msg
    assert set_mood(gid, "nope").startswith("Unknown")
    state = get_state(gid)
    assert state.mood == "focus"
    assert "focus" in VIBE_SEEDS
    text = format_queue(state)
    assert "Vibes queue" in text
    assert "focus" in text
