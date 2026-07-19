# Discord bot (Chloride)

All-purpose Discord agent for **smol-doc-analyzer**, powered by
[Chloride](https://github.com/S4IL21/chloride) (Coral agent + Discord integration).

Specialty: insurance document analysis. Also: **notes / transcription**, **DJ / vibes**,
and free-form **chat** when you mention the bot or use the `--` prefix.

## Setup

```bash
# from repo root
pip install -e ".[discord]"
# optional voice DJ (also needs ffmpeg on PATH):
pip install -e ".[discord,discord-voice]"
# macOS: brew install ffmpeg

python scripts/setup_env.py
# edit .env:
#   DISCORD_TOKEN=...          # Discord bot token (interactive Chloride agent)
#   DISCORD_WEBHOOK_URL=...    # optional inbound webhook for outbound posts
#   OPENROUTER_API_KEY=...     # used as the bot LLM via OpenRouter
# optional:
#   DISCORD_AI_API_KEY=...     # override LLM key
#   DISCORD_AI_MODEL=...       # override model slug
#   OPENAI_API_KEY=...         # preferred for /transcribe (Whisper)

cp discord/smol-doc-analyzer/config.yaml.example discord/smol-doc-analyzer/config.yaml
# replace admin/analyst snowflake IDs in config.yaml
```

Create a Discord application + bot at https://discord.com/developers/applications,
enable **Message Content Intent** (Bot → Privileged Gateway Intents), invite the bot
to your server, and paste the token into `.env` as `DISCORD_TOKEN`.

This integration requests Message Content + voice state (for DJ), not Presence /
Server Members. Invite URL pattern:

`https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&permissions=274878295040&scope=bot%20applications.commands`

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

Mention the bot or prefix a message with `--` (configurable via `DISCORD_PREFIX`)
for free-form chat, **or use slash commands** (synced on bot startup):

| Command | What it does |
|---------|----------------|
| `/analyze` | Run the pipeline on pasted `text` and/or a PDF/PNG `attachment` |
| `/analyze_url` | Download a document from a URL and analyze it |
| `/note add\|list\|search\|show\|delete` | Capture and retrieve server notes |
| `/transcribe` | Transcribe a voice note / audio attachment (optional save to notes) |
| `/play` `/queue` `/skip` `/stop` | DJ queue (voice when deps present; else link queue) |
| `/join` `/leave` | Voice channel control |
| `/vibe` | Set mood: `focus` · `chill` · `energy` · `jazz` · `claims` |
| `/poll` | Quick reaction poll |
| `/remind` | Save a reminder note |
| `/status` | Secret/config + voice-deps readiness (never prints secret values) |
| `/help` | List slash commands |
| `/ping` | Gateway latency check |

### Document pipeline

`/analyze` and the Chloride tool `analyze_insurance_document` call the **memo
chain** in `src/pipeline/`:

`to_markdown → classify → extract → vision_llm → summarize`

The paper Fig. 1 DICIE path (`src/docie/` — medical bills / salvage claims) is
a separate CLI / optional FastAPI entry point; it is not wired to Discord slash
commands. See [src/docie/README.md](../../src/docie/README.md) and
[docs/docie_pipeline.md](../../docs/docie_pipeline.md).

### Notes & transcription

Notes live in `data/discord/notes.db` (gitignored). Transcription prefers
`OPENAI_API_KEY` + Whisper; otherwise tries OpenRouter-compatible STT with
`OPENROUTER_API_KEY`.

### DJ / vibes

- With `ffmpeg` + `PyNaCl` + `yt-dlp`: joins voice and plays audio from YouTube/search.
- Without those deps: queues shareable links so the room can still keep the vibes.

### Chat agent tools

| Surface | Purpose |
|---------|---------|
| `/analyze`, `/analyze_url` | Direct slash → local pipeline |
| `/note`, `/transcribe`, `/remind` | Notes + STT |
| `/play` … `/vibe` | DJ / vibes |
| `/poll`, `/status`, `/help`, `/ping` | Server utilities |
| Agent tool `analyze_insurance_document` | Pipeline via Chloride chat |
| Agent tools `save_note`, `search_notes`, `transcribe_audio`, `vibe_control`, `server_help` | Same capabilities via chat |
| Chloride built-ins | `analyse_file`, search, shell/code (tier-gated) |
| Context menu **Ask Me** | Analyze a selected message |

Tier `allowed_tools` in `config.yaml` controls who may call which tools. The
`default` tier can use docs, notes, STT, vibes, and search; shell/code/reboot stay
admin-only via `*`.

## Run (Docker)

Optional Compose stack (bot process only; models still load from the mounted repo):

```bash
cd discord/smol-doc-analyzer
cp config.yaml.example config.yaml   # if not already
docker compose up --build
```

Voice DJ inside Docker also needs `ffmpeg` in the image and host device access;
link-queue mode works without that.

## Secrets

Never commit `config.yaml` or `memory.db`. Prefer `.env` for `DISCORD_TOKEN` and
API keys; the runner overlays env secrets onto placeholder config values.
