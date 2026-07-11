"""Tests for Discord slash-command helpers (no live Discord required)."""

from __future__ import annotations

import asyncio
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


def test_status_message_has_no_secret_values(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-should-not-appear")
    monkeypatch.setenv("DISCORD_TOKEN", "discord-secret-should-not-appear")
    text = _format_status_message()
    assert "sk-secret-should-not-appear" not in text
    assert "discord-secret-should-not-appear" not in text
    assert "DISCORD_TOKEN" in text
    assert "yes" in text


def test_register_slash_commands_on_mock_tree():
    pytest.importorskip("discord")

    class FakeTree:
        def __init__(self):
            self._commands = {}

        def command(self, *, name, description):
            def deco(fn):
                self._commands[name] = fn
                fn.name = name
                return fn

            return deco

        def get_commands(self):
            return list(self._commands.values())

    bot = SimpleNamespace(tree=FakeTree(), latency=0.042, _may_chat=lambda user: (True, None))
    register_slash_commands(bot)
    names = {c.name for c in bot.tree.get_commands()}
    assert names >= {"analyze", "analyze_url", "status", "help", "ping"}
    register_slash_commands(bot)
    assert len(bot.tree.get_commands()) == 5


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
