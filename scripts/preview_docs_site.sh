#!/usr/bin/env bash
# Preview the Quarto documentation website (docs/).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v quarto >/dev/null 2>&1; then
  echo "error: Quarto CLI not found on PATH." >&2
  echo "Install from https://quarto.org/docs/get-started/ then re-run." >&2
  exit 1
fi

if [[ ! -f docs/_quarto.yml ]]; then
  echo "error: docs/_quarto.yml missing; is this the smol-doc-analyzer repo?" >&2
  exit 1
fi

echo "Starting Quarto preview for docs/ (Ctrl+C to stop)..."
exec quarto preview docs "$@"
