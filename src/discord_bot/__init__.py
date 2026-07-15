"""Discord front-end for smol-doc-analyzer, powered by Chloride (Coral).

Chloride: https://github.com/S4IL21/chloride
"""

from __future__ import annotations

__all__ = ["register_tools", "register_slash_commands", "run_bot"]


def register_tools() -> None:
    """Attach pipeline tools to the Chloride agent (no-op if Chloride missing)."""
    from src.discord_bot.tools import register_tools as _register

    try:
        _register()
    except ImportError:
        # Allow callers / tests without the discord extra installed.
        pass


def register_slash_commands(bot) -> None:
    """Attach Discord app (/) commands onto a Chloride CoralBot instance."""
    from src.discord_bot.commands import register_slash_commands as _register

    _register(bot)


def run_bot(config_dir: str | None = None) -> None:
    from src.discord_bot.runner import main

    main(config_dir=config_dir)
