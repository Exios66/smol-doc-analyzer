"""
Shared pytest fixtures and import hygiene.

The repo keeps Chloride bot assets under ``discord/smol-doc-analyzer/``. That
directory becomes a namespace package named ``discord`` when the repo root is
on ``sys.path``, which shadows the real ``discord.py`` dependency. Tests that
need ``discord.app_commands`` get a lightweight stub when the real package is
unavailable.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _install_discord_stub() -> None:
    """Install a minimal discord.py-compatible stub into sys.modules."""
    discord_stub = types.ModuleType("discord")
    discord_stub.__file__ = str(Path(__file__).resolve())
    discord_stub.__path__ = []  # type: ignore[attr-defined]
    discord_stub.__version__ = "0.0.0-test-stub"

    app_commands = types.ModuleType("discord.app_commands")

    class CommandTree:
        def command(self, *args: Any, **kwargs: Any):
            def decorator(fn):
                fn.name = kwargs.get("name", getattr(fn, "__name__", "cmd"))
                return fn

            return decorator

        def add_command(self, *args: Any, **kwargs: Any) -> None:
            return None

    class Choice:
        def __init__(self, name: str, value: str):
            self.name = name
            self.value = value

    class Group:
        def __init__(self, *, name: str, description: str = ""):
            self.name = name
            self.description = description
            self._commands: dict[str, Any] = {}

        def command(self, *args: Any, **kwargs: Any):
            def decorator(fn):
                fn.name = kwargs.get("name", getattr(fn, "__name__", "cmd"))
                self._commands[fn.name] = fn
                return fn

            return decorator

    class _RangeMeta(type):
        def __getitem__(cls, item):
            return int

    class Range(Generic[T], metaclass=_RangeMeta):
        pass

    def describe(**kwargs: Any):
        def decorator(fn):
            return fn

        return decorator

    def choices(*args: Any, **kwargs: Any):
        def decorator(fn):
            return fn

        return decorator

    def command(*args: Any, **kwargs: Any):
        def decorator(fn):
            fn.name = kwargs.get("name", getattr(fn, "__name__", "cmd"))
            return fn

        return decorator

    app_commands.CommandTree = CommandTree
    app_commands.Choice = Choice
    app_commands.Group = Group
    app_commands.Range = Range
    app_commands.describe = describe
    app_commands.choices = choices
    app_commands.command = command

    class Intents:
        @staticmethod
        def default():
            obj = Intents()
            return obj

        def __init__(self) -> None:
            self.message_content = True
            self.guilds = True
            self.messages = True
            self.voice_states = False

    class Interaction:
        pass

    class Attachment:
        filename: str = "file.bin"
        content_type: str | None = None
        size: int = 0

        async def save(self, path: Path) -> None:
            Path(path).write_bytes(b"")

    class Client:
        def __init__(self, *args: Any, **kwargs: Any):
            self.tree = CommandTree()

        def event(self, fn):
            return fn

    discord_stub.Intents = Intents
    discord_stub.Interaction = Interaction
    discord_stub.Attachment = Attachment
    discord_stub.Client = Client
    discord_stub.Object = lambda *a, **k: object()
    discord_stub.app_commands = app_commands

    sys.modules["discord"] = discord_stub
    sys.modules["discord.app_commands"] = app_commands


def _ensure_discord_app_commands() -> None:
    local_discord = Path(__file__).resolve().parents[1] / "discord"
    existing = sys.modules.get("discord")

    # Drop namespace-only module created from the local discord/ directory.
    if existing is not None and getattr(existing, "__file__", None) is None:
        paths = [Path(p) for p in (getattr(existing, "__path__", None) or [])]
        if any(p == local_discord for p in paths):
            del sys.modules["discord"]
            for key in list(sys.modules):
                if key.startswith("discord."):
                    del sys.modules[key]

    try:
        import discord
        from discord import app_commands  # noqa: F401

        # If import still resolves to the local namespace, replace with stub.
        paths = [Path(p) for p in (getattr(discord, "__path__", None) or [])]
        if getattr(discord, "__file__", None) is None and any(p == local_discord for p in paths):
            _install_discord_stub()
            return
        if not hasattr(discord, "app_commands"):
            _install_discord_stub()
    except Exception:
        _install_discord_stub()


_ensure_discord_app_commands()
