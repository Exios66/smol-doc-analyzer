#!/usr/bin/env bash
# Preview the Quarto documentation website (docs/).
# Run from the docs/ directory so Quarto does not treat the repo-root
# .env.example as a dotenv-safe template (empty secrets would fail).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS="$ROOT/docs"

if ! command -v quarto >/dev/null 2>&1; then
  echo "error: Quarto CLI not found on PATH." >&2
  echo "Install from https://quarto.org/docs/get-started/ then re-run." >&2
  exit 1
fi

if [[ ! -f "$DOCS/_quarto.yml" ]]; then
  echo "error: docs/_quarto.yml missing; is this the smol-doc-analyzer repo?" >&2
  exit 1
fi

cd "$DOCS"
echo "Starting Quarto preview for docs/ (Ctrl+C to stop)..."
exec quarto preview "$@"
