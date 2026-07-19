#!/usr/bin/env bash
# Launch wrapper for the Chloride Discord bot (used by macOS LaunchAgent).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: missing venv python at ${PYTHON}" >&2
  echo "Create it with: python3.11 -m venv .venv && .venv/bin/pip install -e '.[discord]'" >&2
  exit 1
fi

# Ensure Homebrew tools (e.g. ffmpeg for voice DJ) are on PATH for launchd.
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

# Load secrets from repo .env (python-dotenv also loads inside the runner).
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

exec "$PYTHON" -m src.discord_bot --config-dir "${REPO_ROOT}/discord/smol-doc-analyzer"
