"""Discord front-end for smol-doc-analyzer, powered by Chloride (Coral).

Chloride: https://github.com/S4IL21/chloride
"""

from __future__ import annotations

__all__ = ["register_tools", "run_bot"]


def register_tools() -> None:
    """Import side-effects that attach pipeline tools to the Chloride agent."""
    from src.discord_bot import tools as _tools  # noqa: F401


def run_bot(config_dir: str | None = None) -> None:
    from src.discord_bot.runner import main

    main(config_dir=config_dir)
