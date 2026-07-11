# Discord bot (Chloride)

Discord front-end for **smol-doc-analyzer**, powered by
[Chloride](https://github.com/S4IL21/chloride) (Coral agent + Discord integration).

## Setup

```bash
# from repo root
pip install -e ".[discord]"
python scripts/setup_env.py
# edit .env:
#   DISCORD_TOKEN=...          # Discord bot token (interactive Chloride agent)
#   DISCORD_WEBHOOK_URL=...    # optional inbound webhook for outbound posts
#   OPENROUTER_API_KEY=...     # used as the bot LLM via OpenRouter
# optional:
#   DISCORD_AI_API_KEY=...     # override LLM key
#   DISCORD_AI_MODEL=...       # override model slug

cp discord/smol-doc-analyzer/config.yaml.example discord/smol-doc-analyzer/config.yaml
# replace admin/analyst snowflake IDs in config.yaml
```

Create a Discord application + bot at https://discord.com/developers/applications,
enable **Message Content Intent**, invite the bot to your server, and paste the token
into `.env` as `DISCORD_TOKEN`.

For **outbound-only** notifications (no interactive replies), create a channel webhook
(Channel settings → Integrations → Webhooks) and set `DISCORD_WEBHOOK_URL` in `.env`.
A webhook cannot replace the Chloride bot — it only posts messages.

## Webhook notifications (outbound)

```bash
python -m src.discord_bot.webhook --check
python -m src.discord_bot.webhook --text "LOSS NOTICE
Claim Number: CLM-1
Date of Loss: 2024-01-15
Loss Type: collision"
python -m src.discord_bot.webhook --pdf path/to/claim.pdf
```

Never commit webhook URLs. If one was pasted into chat or a ticket, rotate it in
Discord (Edit Webhook → Reset Token / delete & recreate).

## Run (local)

```bash
python -m src.discord_bot
# or:
python -m src.discord_bot --config-dir discord/smol-doc-analyzer
```

Mention the bot or prefix a message with `--` (configurable via `DISCORD_PREFIX`).

Attach a PDF/PNG or paste document text, then ask it to analyze — the agent should
call `analyze_insurance_document`, which runs:

`to_markdown → classify → extract → vision_llm → summarize`

## Run (Docker)

Optional Compose stack (bot process only; models still load from the mounted repo):

```bash
cd discord/smol-doc-analyzer
cp config.yaml.example config.yaml   # if not already
docker compose up --build
```

## Tools

| Tool | Purpose |
|------|---------|
| `analyze_insurance_document` | Local pipeline on text / Discord attachment / URL |
| Chloride built-ins | `analyse_file`, search, shell/code (tier-gated) |

Tier `allowed_tools` in `config.yaml` controls who may call which tools. The
`default` tier is limited to document analysis + `get_user_info`.

## Secrets

Never commit `config.yaml` or `memory.db`. Prefer `.env` for `DISCORD_TOKEN` and
API keys; the runner overlays env secrets onto placeholder config values.
