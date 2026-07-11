#!/usr/bin/env python3
"""Create a local `.env` from `.env.example` and report secret readiness.

Does not print secret values. Safe to re-run; never overwrites an existing `.env`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local .env for secrets")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .env from .env.example (destructive)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only print which secrets are present (no file creation)",
    )
    args = parser.parse_args()

    example = REPO_ROOT / ".env.example"
    target = REPO_ROOT / ".env"

    if not args.status:
        if not example.exists():
            print(f"missing template: {example}", file=sys.stderr)
            return 1
        if target.exists() and not args.force:
            print(f".env already exists at {target} (pass --force to replace)")
        else:
            shutil.copyfile(example, target)
            print(f"wrote {target}")
            print(
                "Edit that file and paste OPENROUTER_API_KEY / WANDB_API_KEY / "
                "HF_TOKEN / DISCORD_TOKEN (Discord optional)."
            )

    # Import after copy so Config can see the new file
    sys.path.insert(0, str(REPO_ROOT))
    from src.utils.config import secrets_status

    status = secrets_status()
    print("")
    print("Secret readiness (values never shown):")
    for key, present in status.items():
        if key == "dotenv_file":
            print(f"  .env file          {'yes' if present else 'MISSING'}")
        else:
            print(f"  {key:22s} {'set' if present else 'not set'}")

    print("")
    print("Where to get keys:")
    print("  OPENROUTER_API_KEY  https://openrouter.ai/keys")
    print("  WANDB_API_KEY       https://wandb.ai/authorize")
    print("  HF_TOKEN            https://huggingface.co/settings/tokens  (optional)")
    print("  DISCORD_TOKEN       https://discord.com/developers/applications  (optional bot)")
    print("")
    print("Discord bot (Chloride): pip install -e \".[discord]\" then")
    print("  python -m src.discord_bot")
    print("  See discord/smol-doc-analyzer/README.md")
    print("")
    print("Cursor Cloud Agents: add the same keys as Environment Secrets in the")
    print("Cursor dashboard for this repo/environment so cloud runs can use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
