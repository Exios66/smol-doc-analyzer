#!/usr/bin/env bash
# Fallback autostart via macOS Login Items (uses Terminal, which can read Desktop).
# Prefer ./scripts/install_discord_bot_launchagent.sh after granting Full Disk Access
# to /bin/bash — this Login Item is the workaround when the repo lives on Desktop.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND_PATH="${REPO_ROOT}/scripts/Start Chloride Bot.command"

chmod +x "$COMMAND_PATH" "${REPO_ROOT}/scripts/run_discord_bot.sh"

# Remove any prior login item with the same path, then add.
osascript <<EOF
tell application "System Events"
  set wanted to POSIX file "${COMMAND_PATH}"
  set existing to login items
  repeat with i in existing
    try
      if path of i is (POSIX path of wanted) then
        delete i
      end if
    end try
  end repeat
  make login item at end with properties {path:"${COMMAND_PATH}", hidden:true}
end tell
EOF

echo "Added Login Item: ${COMMAND_PATH}"
echo "It will start the Chloride bot when you sign in (hidden)."
echo
echo "Remove later with:"
echo "  ./scripts/uninstall_discord_bot_loginitem.sh"
echo "  # or System Settings → General → Login Items"
