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

# Concurrent quarto render/preview can corrupt Deno KV SassCache
# (ERROR: BadResource: Bad resource ID). Clear local project cache when
# requested, or when a previous preview left a broken cache behind.
if [[ "${QUARTO_CLEAN:-}" == "1" ]] || [[ "${1:-}" == "--clean" ]]; then
  if [[ "${1:-}" == "--clean" ]]; then
    shift
  fi
  echo "Cleaning Quarto project + Sass caches..."
  rm -rf "$DOCS/.quarto" "$HOME/Library/Caches/quarto/sass"
fi

cd "$DOCS"
echo "Starting Quarto preview for docs/ (Ctrl+C to stop)..."
echo "Tip: QUARTO_CLEAN=1 ./scripts/preview_docs_site.sh  # or --clean"
exec quarto preview "$@"
