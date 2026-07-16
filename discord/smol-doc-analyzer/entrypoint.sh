#!/usr/bin/env sh
# Bootstrap Chloride config from the example when config.yaml is absent.
set -eu
CONFIG_DIR="${1:-discord/smol-doc-analyzer}"
EXAMPLE="${CONFIG_DIR}/config.yaml.example"
TARGET="${CONFIG_DIR}/config.yaml"

if [ ! -f "${TARGET}" ] && [ -f "${EXAMPLE}" ]; then
  cp "${EXAMPLE}" "${TARGET}"
  echo "Created ${TARGET} from example (set Discord/API secrets via env)."
fi

exec python -m src.discord_bot --config-dir "${CONFIG_DIR}"
