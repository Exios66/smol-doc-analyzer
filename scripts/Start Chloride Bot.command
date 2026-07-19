#!/usr/bin/env bash
# Double-click / Login Item launcher (runs with Terminal's Desktop access).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/data/discord/logs"
mkdir -p "$LOG_DIR"

# If LaunchAgent already owns the process, do nothing.
if pgrep -f "${REPO_ROOT}/.venv/bin/python -m src.discord_bot" >/dev/null 2>&1; then
  echo "Chloride bot already running."
  sleep 1
  exit 0
fi

nohup "${REPO_ROOT}/scripts/run_discord_bot.sh" \
  >>"${LOG_DIR}/discord-bot.stdout.log" \
  2>>"${LOG_DIR}/discord-bot.stderr.log" &

echo "Started Chloride Discord bot (pid $!)."
sleep 1
exit 0
