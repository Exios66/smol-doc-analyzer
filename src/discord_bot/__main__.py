"""CLI entry: python -m src.discord_bot"""

from __future__ import annotations

import argparse

from src.discord_bot.runner import DEFAULT_CONFIG_DIR, main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Chloride Discord bot with smol-doc-analyzer pipeline tools. "
            "Upstream: https://github.com/S4IL21/chloride"
        )
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help=f"Bot workspace directory (default: {DEFAULT_CONFIG_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(config_dir=args.config_dir)
