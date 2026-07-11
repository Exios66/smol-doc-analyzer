"""Boot the Chloride Discord bot with smol-doc-analyzer tools registered."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
import yaml

from src.utils.config import REPO_ROOT, _load_dotenv, _secret

DEFAULT_CONFIG_DIR = REPO_ROOT / "discord" / "smol-doc-analyzer"


def _resolve_config_dir(config_dir: str | Path | None) -> Path:
    if config_dir is None:
        env = os.getenv("DISCORD_BOT_CONFIG_DIR", "").strip()
        path = Path(env) if env else DEFAULT_CONFIG_DIR
    else:
        path = Path(config_dir)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _overlay_secrets(raw: dict) -> dict:
    """Fill token/API key placeholders from environment / .env."""
    data = dict(raw)
    discord_token = _secret("DISCORD_TOKEN") or os.getenv("DISCORD_TOKEN", "").strip()
    ai_key = (
        _secret("DISCORD_AI_API_KEY")
        or _secret("AI_API_KEY")
        or _secret("OPENROUTER_API_KEY")
        or os.getenv("AI_API_KEY", "").strip()
    )

    token = data.get("DISCORD_TOKEN") or ""
    if not token or "paste" in str(token).lower() or "your" in str(token).lower():
        if discord_token:
            data["DISCORD_TOKEN"] = discord_token

    key = data.get("AI_API_KEY") or ""
    if not key or "put your" in str(key).lower() or "your" in str(key).lower():
        if ai_key:
            data["AI_API_KEY"] = ai_key

    # Prefer OpenRouter as OpenAI-compatible when configured and no base URL set.
    if not data.get("AI_OPENAI_COMPATIBLE_BASE_URL") and _secret("OPENROUTER_API_KEY"):
        if os.getenv("DISCORD_USE_OPENROUTER", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            data["AI_OPENAI_COMPATIBLE_BASE_URL"] = "https://openrouter.ai/api/v1"
            model = str(data.get("AI_MODEL_NAME") or "")
            # Chloride stock pydantic-ai schemes are not OpenRouter slugs.
            chloride_native = model.startswith(
                ("google-gla:", "google-vertex:", "openai:", "anthropic:", "xai:", "groq:")
            ) and "/" not in model.split(":", 1)[-1]
            if chloride_native or not model:
                data["AI_MODEL_NAME"] = os.getenv(
                    "DISCORD_AI_MODEL", "anthropic/claude-sonnet-4.5"
                )

    return data


def load_bot_config(config_dir: Path):
    from coral.config import Config

    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        example = config_dir / "config.yaml.example"
        raise FileNotFoundError(
            f"Missing {config_path}. Copy {example.name} → config.yaml and edit secrets, "
            "or set DISCORD_TOKEN / AI_API_KEY (or OPENROUTER_API_KEY) in .env."
        )
    raw = yaml.full_load(config_path.read_text(encoding="utf-8")) or {}
    raw = _overlay_secrets(raw)
    return Config.model_validate(raw)


def main(config_dir: str | Path | None = None) -> None:
    """Start the Discord bot (local process; Docker optional via discord workspace)."""
    _load_dotenv()
    cfg_dir = _resolve_config_dir(config_dir)
    if not cfg_dir.exists():
        typer.secho(f"Config directory not found: {cfg_dir}", fg="red")
        raise SystemExit(1)

    # Chloride prompts load AI_EXTRA_CONTEXT_PATH relative to CWD.
    os.chdir(cfg_dir)

    try:
        import src.discord_bot.tools  # noqa: F401 — registers analyze_insurance_document
        from coral.agent import agent
        from coral.bot import CoralBot
        from coral.history import init_db
    except ImportError as exc:
        typer.secho(
            "Chloride Discord dependencies missing. Install with:\n"
            '  pip install -e ".[discord]"\n'
            f"Detail: {exc}",
            fg="red",
        )
        raise SystemExit(1) from exc

    config = load_bot_config(cfg_dir)

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if config.AI_OPENAI_COMPATIBLE_BASE_URL:
        model = OpenAIChatModel(
            config.AI_MODEL_NAME,
            provider=OpenAIProvider(
                base_url=config.AI_OPENAI_COMPATIBLE_BASE_URL,
                api_key=config.AI_API_KEY
                or os.getenv("AI_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
                or "X",
            ),
            settings=config.AI_EXTRA_CONFIG or None,
        )
    else:
        model = config.AI_MODEL_NAME
        # Mirror Chloride core.py provider env wiring for native pydantic-ai models.
        if config.AI_API_KEY and ":" in str(model):
            prefix = str(model).split(":", 1)[0].split("-", 1)[0].upper() + "_API_KEY"
            os.environ[prefix] = config.AI_API_KEY

    engine = init_db(config.DB_PATH)

    import discord

    # Prefer minimal intents. Chloride upstream uses Intents.all(), which also
    # requires Presence + Server Members privileged intents. We only need
    # Message Content (+ guild message events) for analyze_insurance_document.
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.guild_messages = True
    intents.dm_messages = True

    token = config.DISCORD_TOKEN or os.getenv("DISCORD_TOKEN")
    if not token:
        typer.secho(
            "DISCORD_TOKEN not found in config.yaml or environment. "
            "Set it in discord/smol-doc-analyzer/config.yaml or .env.",
            fg="red",
        )
        raise SystemExit(1)

    def _make_client(active_intents: discord.Intents) -> "CoralBot":
        return CoralBot(
            config=config,
            agent=agent,
            model=model,
            intents=active_intents,
            engine=engine,
        )

    typer.secho(
        f"Starting Chloride Discord bot from {cfg_dir} "
        f"(tools include analyze_insurance_document)…",
        fg="green",
    )

    try:
        client = _make_client(intents)
        client.run(token)
    except discord.errors.PrivilegedIntentsRequired:
        typer.secho(
            "\nPrivileged intents are not enabled for this bot application.\n"
            "Open the Discord Developer Portal → your app → Bot → Privileged Gateway Intents\n"
            "and enable **Message Content Intent**, then restart.\n"
            f"Direct link: https://discord.com/developers/applications/"
            f"{_client_id_from_token(token)}/bot\n"
            "Falling back to non-privileged intents so the process stays online "
            "(message text will be empty until Message Content Intent is enabled).",
            fg="yellow",
        )
        fallback = discord.Intents.default()
        fallback.guilds = True
        fallback.guild_messages = True
        fallback.dm_messages = True
        client = _make_client(fallback)
        client.run(token)


def _client_id_from_token(token: str) -> str:
    """Best-effort bot application id from the token's first segment (base64 user id)."""
    import base64

    try:
        part = token.split(".", 1)[0]
        pad = "=" * (-len(part) % 4)
        return base64.b64decode(part + pad).decode("utf-8")
    except Exception:
        return "YOUR_APP_ID"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
