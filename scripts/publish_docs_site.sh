#!/usr/bin/env bash
# Publish the Quarto docs site from this machine (no GitHub Actions).
#
# Preferred host: Posit Connect Cloud (successor to Quarto Pub)
#   ./scripts/publish_docs_site.sh
#   ./scripts/publish_docs_site.sh posit-connect-cloud
#
# Retry upload after a timeout (skip re-render — site already in docs/_site):
#   ./scripts/publish_docs_site.sh --retry
#   QUARTO_PUBLISH_RETRIES=5 ./scripts/publish_docs_site.sh --retry
#
# Legacy Quarto Pub (still works for existing accounts):
#   ./scripts/publish_docs_site.sh quarto-pub
#
# First run opens a browser to authorize your Posit / Quarto account.
# Destination is recorded in docs/_publish.yml (safe to commit).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS="$ROOT/docs"
TARGET="posit-connect-cloud"
NO_RENDER=0
RETRIES="${QUARTO_PUBLISH_RETRIES:-3}"
RETRY_SLEEP="${QUARTO_PUBLISH_RETRY_SLEEP:-8}"

usage() {
  cat <<EOF
usage: $0 [provider] [--retry|--no-render] [--retries N]

  provider   posit-connect-cloud (default) | quarto-pub | netlify
  --retry    Re-upload docs/_site without re-rendering (use after upload timeout)
  --no-render
             Same as --retry
  --retries N
             Upload attempts (default: ${RETRIES}; env QUARTO_PUBLISH_RETRIES)

Examples:
  $0
  $0 --retry
  QUARTO_PUBLISH_NO_BROWSER=1 $0 --retry --retries 5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --retry|--no-render)
      NO_RENDER=1
      shift
      ;;
    --retries)
      RETRIES="${2:?--retries requires a number}"
      shift 2
      ;;
    posit-connect-cloud|quarto-pub|netlify)
      TARGET="$1"
      shift
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v quarto >/dev/null 2>&1; then
  echo "error: Quarto CLI not found on PATH." >&2
  echo "Install from https://quarto.org/docs/get-started/ then re-run." >&2
  exit 1
fi

if [[ ! -f "$DOCS/_quarto.yml" ]]; then
  echo "error: docs/_quarto.yml missing." >&2
  exit 1
fi

if [[ "$NO_RENDER" == "1" && ! -d "$DOCS/_site" ]]; then
  echo "error: --retry/--no-render needs an existing docs/_site (render once first)." >&2
  exit 1
fi

cd "$DOCS"

PUBLISH_ARGS=(publish "$TARGET" --no-prompt)
if [[ "$NO_RENDER" == "1" ]]; then
  PUBLISH_ARGS+=(--no-render)
fi
if [[ "${QUARTO_PUBLISH_NO_BROWSER:-}" == "1" ]]; then
  PUBLISH_ARGS+=(--no-browser)
fi

SITE_SIZE="$(du -sh "$DOCS/_site" 2>/dev/null | awk '{print $1}' || echo "?")"
echo "Publishing docs/ via: quarto ${PUBLISH_ARGS[*]}"
echo "Site bundle: ${SITE_SIZE} (docs/_site)"
if [[ "$NO_RENDER" == "1" ]]; then
  echo "Mode: upload-only (--no-render). Skipping Quarto render."
else
  echo "Mode: render + upload."
fi
echo "Log into the correct Posit / Quarto account in your browser first."
echo "(Ctrl+C to abort.)"
echo

# Soft connectivity hint — failures here are non-fatal.
if command -v curl >/dev/null 2>&1; then
  if ! curl -fsS --connect-timeout 8 --max-time 15 -o /dev/null \
      https://intake.connect.posit.cloud/ 2>/dev/null; then
    echo "warn: cannot reach https://intake.connect.posit.cloud/ quickly." >&2
    echo "      VPN / firewall / Posit outages often cause 'connection error: timed out'." >&2
    echo "      Check https://status.posit.co/ then re-run with: $0 --retry" >&2
    echo
  fi
fi

attempt=1
while (( attempt <= RETRIES )); do
  echo "── publish attempt ${attempt}/${RETRIES} ──"
  set +e
  quarto "${PUBLISH_ARGS[@]}"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    echo
    echo "Publish succeeded."
    exit 0
  fi

  if (( attempt == RETRIES )); then
    echo
    echo "error: publish failed after ${RETRIES} attempt(s) (exit ${status})." >&2
    echo >&2
    echo "If you saw 'Uploading files' then 'connection error: timed out':" >&2
    echo "  • Auth + render already worked — only the Posit intake upload failed." >&2
    echo "  • Site size is small (~8MB); this is almost always network/VPN/Posit-side." >&2
    echo "  • Retry without re-rendering:" >&2
    echo "      $0 --retry" >&2
    echo "  • Or try off VPN / another network, then check https://status.posit.co/" >&2
    exit "$status"
  fi

  echo
  echo "warn: attempt ${attempt} failed (exit ${status}); retrying in ${RETRY_SLEEP}s…" >&2
  echo "      Tip: subsequent attempts use --no-render so you do not wait on notebooks again." >&2
  NO_RENDER=1
  PUBLISH_ARGS=(publish "$TARGET" --no-prompt --no-render)
  if [[ "${QUARTO_PUBLISH_NO_BROWSER:-}" == "1" ]]; then
    PUBLISH_ARGS+=(--no-browser)
  fi
  sleep "$RETRY_SLEEP"
  attempt=$((attempt + 1))
done
