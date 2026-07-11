"""Post outbound Discord messages via an incoming webhook URL.

Webhooks are one-way (bot → channel). The interactive Chloride agent still needs
DISCORD_TOKEN. Never commit webhook URLs; load them from .env only.

CLI:
  python -m src.discord_bot.webhook --check
  python -m src.discord_bot.webhook --text "LOSS NOTICE\\nClaim Number: CLM-1\\n..."
  python -m src.discord_bot.webhook --pdf path/to/claim.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Any

from src.discord_bot.formatters import compact_analysis, format_discord_summary
from src.utils.config import _load_dotenv, _secret

logger = logging.getLogger(__name__)

# Discord allows up to 2000 chars per content field.
_MAX_CONTENT = 1900


def webhook_url() -> str:
    """Return DISCORD_WEBHOOK_URL from env/.env, or empty if unset."""
    _load_dotenv()
    return _secret("DISCORD_WEBHOOK_URL")


def post_webhook(
    content: str,
    *,
    url: str | None = None,
    username: str | None = "smol-doc-analyzer",
    embeds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    POST a message to a Discord incoming webhook.

    Returns a small status dict; never includes the webhook URL.
    """
    target = (url or webhook_url()).strip()
    if not target:
        return {"ok": False, "error": "DISCORD_WEBHOOK_URL is not set"}
    if (
        "discord.com/api/webhooks/" not in target
        and "discordapp.com/api/webhooks/" not in target
    ):
        return {
            "ok": False,
            "error": "DISCORD_WEBHOOK_URL does not look like a Discord webhook",
        }

    body: dict[str, Any] = {"content": (content or "")[:_MAX_CONTENT]}
    if username:
        body["username"] = username
    if embeds:
        body["embeds"] = embeds[:10]

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        target,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "smol-doc-analyzer-discord-webhook/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            status = getattr(resp, "status", None) or resp.getcode()
            # Discord returns 204 No Content on success for plain webhooks.
            return {"ok": 200 <= int(status) < 300, "status": int(status)}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        logger.error("Discord webhook HTTP %s: %s", exc.code, detail)
        return {"ok": False, "status": exc.code, "error": detail}
    except Exception as exc:
        logger.exception("Discord webhook request failed")
        return {"ok": False, "error": str(exc)}


def post_analysis(
    result: dict[str, Any],
    *,
    url: str | None = None,
    username: str | None = "smol-doc-analyzer",
) -> dict[str, Any]:
    """Post a compact analysis summary to the configured Discord webhook."""
    compact = compact_analysis(result)
    content = format_discord_summary(compact)
    return post_webhook(content, url=url, username=username)


def post_connection_check(*, url: str | None = None) -> dict[str, Any]:
    """Send a short connectivity probe (useful after configuring DISCORD_WEBHOOK_URL)."""
    return post_webhook(
        "**smol-doc-analyzer** webhook connected.\n"
        "- Outbound notifications: ready\n"
        "- Interactive Chloride bot still needs `DISCORD_TOKEN` separately",
        url=url,
    )


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Post messages or pipeline analysis results to DISCORD_WEBHOOK_URL"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Send a short connectivity probe to the webhook",
    )
    parser.add_argument("--message", type=str, default=None, help="Post arbitrary text")
    parser.add_argument("--text", type=str, default=None, help="Analyze text then post summary")
    parser.add_argument("--pdf", type=str, default=None, help="Analyze PDF then post summary")
    parser.add_argument("--image", type=str, default=None, help="Analyze image then post summary")
    parser.add_argument("--vision", action="store_true", help="Enable vision stage for analysis")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision stage")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable status (never includes the webhook URL)",
    )
    args = parser.parse_args(argv)

    if not webhook_url():
        msg = "DISCORD_WEBHOOK_URL is not set. Add it to .env (gitignored) and retry."
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(msg, file=sys.stderr)
        return 1

    if args.no_vision:
        enable_vision: bool | None = False
    elif args.vision:
        enable_vision = True
    else:
        enable_vision = False

    if args.check:
        result = post_connection_check()
    elif args.message is not None:
        result = post_webhook(args.message)
    elif args.text is not None or args.pdf or args.image:
        from src.pipeline.orchestrator import analyze_document

        analysis = analyze_document(
            args.text or "",
            record_id="discord-webhook",
            pdf_path=args.pdf,
            image_path=args.image,
            enable_vision=enable_vision,
        )
        result = post_analysis(analysis)
        result = {**result, "record_id": analysis.get("record_id")}
    else:
        parser.error("Provide --check, --message, or --text/--pdf/--image")
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    elif result.get("ok"):
        print(f"Posted to Discord webhook (HTTP {result.get('status', 'ok')}).")
    else:
        print(f"Webhook post failed: {result.get('error') or result}", file=sys.stderr)
        return 1
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
