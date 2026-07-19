#!/usr/bin/env bash
# Install / reload a macOS LaunchAgent so the Chloride bot starts at login
# and restarts if it exits.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.smol-doc-analyzer.discord-bot"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
RUNNER="${REPO_ROOT}/scripts/run_discord_bot.sh"
LOG_DIR="${REPO_ROOT}/data/discord/logs"
UID_NUM="$(id -u)"

chmod +x "$RUNNER"
mkdir -p "$PLIST_DIR" "$LOG_DIR"

# macOS TCC blocks launchd from Desktop/Documents/Downloads unless Full Disk Access
# is granted to /bin/bash (or the repo is moved outside those folders).
NEEDS_FDA=0
case "$REPO_ROOT" in
  "$HOME/Desktop"*|"$HOME/Documents"*|"$HOME/Downloads"*) NEEDS_FDA=1 ;;
esac

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

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>ThrottleInterval</key>
  <integer>15</integer>

  <key>ProcessType</key>
  <string>Background</string>

  <key>StandardOutPath</key>
  <string>${LOG_DIR}/discord-bot.stdout.log</string>

  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/discord-bot.stderr.log</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
</dict>
</plist>
EOF

# Unload if already loaded (ignore errors when not loaded).
launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
launchctl unload "$PLIST_PATH" 2>/dev/null || true

# Prefer modern bootstrap; fall back to load.
if launchctl bootstrap "gui/${UID_NUM}" "$PLIST_PATH" 2>/dev/null; then
  launchctl enable "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
  launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
else
  launchctl load -w "$PLIST_PATH"
fi

sleep 2
EXIT_CODE="$(launchctl print "gui/${UID_NUM}/${LABEL}" 2>/dev/null | awk '/last exit code/ {print $4; exit}' || true)"

echo "Installed LaunchAgent: ${PLIST_PATH}"
echo "Label: ${LABEL}"
echo "Logs:  ${LOG_DIR}/discord-bot.*.log"
echo

if [[ "$NEEDS_FDA" == "1" ]] || [[ "${EXIT_CODE}" == "126" ]]; then
  cat <<'MSG'
⚠  This repo lives under Desktop/Documents/Downloads (or launchd returned 126).
   macOS blocks background LaunchAgents from that folder until you grant access:

   1. System Settings → Privacy & Security → Full Disk Access
   2. Click +, press Cmd+Shift+G, enter:  /bin/bash
   3. Enable the toggle for /bin/bash
   4. Re-run:
        ./scripts/install_discord_bot_launchagent.sh

MSG
  # Open the Full Disk Access pane when possible (macOS Ventura+).
  open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles" 2>/dev/null \
    || open "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles" 2>/dev/null \
    || true
fi

echo "Useful commands:"
echo "  launchctl print gui/\$(id -u)/${LABEL}"
echo "  tail -f ${LOG_DIR}/discord-bot.stdout.log"
echo "  ${REPO_ROOT}/scripts/uninstall_discord_bot_launchagent.sh"
