#!/usr/bin/env bash
# Remove the weekly CHANGELOG LaunchAgent (or crontab entry).
set -euo pipefail

LABEL="com.smol-doc-analyzer.update-changelog"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--crontab" ]]; then
  EXISTING="$(crontab -l 2>/dev/null || true)"
  FILTERED="$(printf '%s\n' "$EXISTING" | grep -v 'scripts/update_changelog.sh' || true)"
  if [[ -n "$FILTERED" ]]; then
    printf '%s\n' "$FILTERED" | crontab -
  else
    # Empty crontab
    crontab -r 2>/dev/null || true
  fi
  echo "Removed crontab entries for scripts/update_changelog.sh"
  exit 0
fi

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl unload "$PLIST_PATH" 2>/dev/null || true

if [[ -f "$PLIST_PATH" ]]; then
  rm -f "$PLIST_PATH"
  echo "Removed ${PLIST_PATH}"
else
  echo "No plist at ${PLIST_PATH}"
fi

echo "LaunchAgent ${LABEL} unloaded."
echo "(Optional) also clear crontab: ${REPO_ROOT}/scripts/uninstall_changelog_launchagent.sh --crontab"
