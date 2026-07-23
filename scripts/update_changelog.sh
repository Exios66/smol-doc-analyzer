#!/usr/bin/env bash
# Wrapper for update_changelog.py — used by LaunchAgent / cron and manual runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TZ="${TZ:-America/Chicago}"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"

LOG_DIR="${REPO_ROOT}/data/changelog/logs"
mkdir -p "$LOG_DIR"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

STAMP="$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "[${STAMP}] update_changelog starting (TZ=${TZ})"

"$PYTHON" "${REPO_ROOT}/scripts/update_changelog.py" "$@"
