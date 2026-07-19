#!/usr/bin/env bash
# Stop and remove the Chloride Discord bot LaunchAgent.
set -euo pipefail

LABEL="com.smol-doc-analyzer.discord-bot"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl unload "$PLIST_PATH" 2>/dev/null || true

if [[ -f "$PLIST_PATH" ]]; then
  rm -f "$PLIST_PATH"
  echo "Removed ${PLIST_PATH}"
else
  echo "No plist at ${PLIST_PATH}"
fi

echo "LaunchAgent ${LABEL} unloaded."
