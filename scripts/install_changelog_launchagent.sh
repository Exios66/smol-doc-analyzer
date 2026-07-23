#!/usr/bin/env bash
# Install a macOS LaunchAgent that refreshes CHANGELOG.md every Wednesday at
# 11:00 PM America/Chicago (CST/CDT). Also supports a user-crontab fallback.
#
# Usage:
#   ./scripts/install_changelog_launchagent.sh           # LaunchAgent (default)
#   ./scripts/install_changelog_launchagent.sh --crontab # user crontab instead
#   ./scripts/update_changelog.sh                        # run once now
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.smol-doc-analyzer.update-changelog"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
RUNNER="${REPO_ROOT}/scripts/update_changelog.sh"
LOG_DIR="${REPO_ROOT}/data/changelog/logs"
UID_NUM="$(id -u)"
MODE="launchd"

if [[ "${1:-}" == "--crontab" ]]; then
  MODE="crontab"
fi

chmod +x "$RUNNER" \
  "${REPO_ROOT}/scripts/update_changelog.py" \
  "${REPO_ROOT}/scripts/uninstall_changelog_launchagent.sh" \
  "${REPO_ROOT}/scripts/install_changelog_launchagent.sh"
mkdir -p "$PLIST_DIR" "$LOG_DIR"

# macOS TCC blocks launchd from Desktop/Documents/Downloads unless Full Disk Access
# is granted to /bin/bash (or the repo is moved outside those folders).
NEEDS_FDA=0
case "$REPO_ROOT" in
  "$HOME/Desktop"*|"$HOME/Documents"*|"$HOME/Downloads"*) NEEDS_FDA=1 ;;
esac

if [[ "$MODE" == "crontab" ]]; then
  # Wednesday 23:00 America/Chicago
  CRON_LINE="0 23 * * 3 TZ=America/Chicago ${RUNNER} >>${LOG_DIR}/cron.log 2>&1"
  EXISTING="$(crontab -l 2>/dev/null || true)"
  FILTERED="$(printf '%s\n' "$EXISTING" | grep -v 'scripts/update_changelog.sh' || true)"
  {
    printf '%s\n' "$FILTERED"
    printf '%s\n' "$CRON_LINE"
  } | sed '/^$/d' | crontab -
  echo "Installed user crontab entry:"
  echo "  ${CRON_LINE}"
  echo "Logs: ${LOG_DIR}/cron.log"
  echo
  echo "Run once now:"
  echo "  ${RUNNER}"
  echo "Remove with:"
  echo "  ${REPO_ROOT}/scripts/uninstall_changelog_launchagent.sh --crontab"
  exit 0
fi

# LaunchAgent: Wednesday (3) at 23:00 local time.
# Set the Mac timezone to America/Chicago so this matches 11pm CST/CDT.
# System Settings → General → Date & Time → Time Zone.
cat >"$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RUNNER}</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key>
    <integer>3</integer>
    <key>Hour</key>
    <integer>23</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>

  <key>RunAtLoad</key>
  <false/>

  <key>ProcessType</key>
  <string>Background</string>

  <key>StandardOutPath</key>
  <string>${LOG_DIR}/changelog.stdout.log</string>

  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/changelog.stderr.log</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>TZ</key>
    <string>America/Chicago</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
launchctl unload "$PLIST_PATH" 2>/dev/null || true

if launchctl bootstrap "gui/${UID_NUM}" "$PLIST_PATH" 2>/dev/null; then
  launchctl enable "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
else
  launchctl load -w "$PLIST_PATH"
fi

echo "Installed LaunchAgent: ${PLIST_PATH}"
echo "Label: ${LABEL}"
echo "Schedule: Wednesday 23:00 (set Mac time zone to America/Chicago for CST/CDT)"
echo "Logs:  ${LOG_DIR}/changelog.*.log"
echo

if [[ "$NEEDS_FDA" == "1" ]]; then
  cat <<'MSG'
⚠  This repo lives under Desktop/Documents/Downloads.
   macOS may block LaunchAgents from that folder until you grant access:

   1. System Settings → Privacy & Security → Full Disk Access
   2. Click +, press Cmd+Shift+G, enter:  /bin/bash
   3. Enable the toggle for /bin/bash
   4. Re-run:
        ./scripts/install_changelog_launchagent.sh

MSG
fi

echo "Useful commands:"
echo "  ${RUNNER}                      # run now"
echo "  ${RUNNER} --dry-run --print-unreleased"
echo "  launchctl print gui/\$(id -u)/${LABEL}"
echo "  launchctl kickstart -k gui/\$(id -u)/${LABEL}   # force a run"
echo "  ${REPO_ROOT}/scripts/uninstall_changelog_launchagent.sh"
