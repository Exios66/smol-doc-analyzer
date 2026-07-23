#!/usr/bin/env bash
# Publish the Quarto docs site from this machine (no GitHub Actions).
#
# Preferred host: Posit Connect Cloud (successor to Quarto Pub)
#   ./scripts/publish_docs_site.sh
#   ./scripts/publish_docs_site.sh posit-connect-cloud
#
# Legacy Quarto Pub (still works for existing accounts):
#   ./scripts/publish_docs_site.sh quarto-pub
#
# First run opens a browser to authorize your Posit / Quarto account.
# Destination is recorded in docs/_publish.yml (safe to commit).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS="$ROOT/docs"
TARGET="${1:-posit-connect-cloud}"

if ! command -v quarto >/dev/null 2>&1; then
  echo "error: Quarto CLI not found on PATH." >&2
  echo "Install from https://quarto.org/docs/get-started/ then re-run." >&2
  exit 1
fi

if [[ ! -f "$DOCS/_quarto.yml" ]]; then
  echo "error: docs/_quarto.yml missing." >&2
  exit 1
fi

case "$TARGET" in
  posit-connect-cloud|quarto-pub|netlify)
    ;;
  *)
    echo "usage: $0 [posit-connect-cloud|quarto-pub|netlify]" >&2
    exit 2
    ;;
esac

cd "$DOCS"
echo "Publishing docs/ via: quarto publish ${TARGET}"
echo "Log into the correct Posit / Quarto account in your browser first."
echo "(First run will prompt to authorize the Quarto CLI. Ctrl+C to abort.)"

# Allow browser for OAuth + post-publish view. Set QUARTO_PUBLISH_NO_BROWSER=1 to skip.
if [[ "${QUARTO_PUBLISH_NO_BROWSER:-}" == "1" ]]; then
  exec quarto publish "$TARGET" --no-browser
else
  exec quarto publish "$TARGET"
fi
