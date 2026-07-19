#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND_PATH="${REPO_ROOT}/scripts/Start Chloride Bot.command"

osascript <<EOF
tell application "System Events"
  set wanted to POSIX path of (POSIX file "${COMMAND_PATH}")
  repeat with i in (get login items)
    try
      if path of i is wanted then
        delete i
      end if
    end try
  end repeat
end tell
EOF

echo "Removed Login Item for: ${COMMAND_PATH}"
