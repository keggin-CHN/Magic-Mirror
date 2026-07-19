#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="${SCRIPT_DIR}/install-server-linux.sh"

if [ ! -f "$INSTALLER" ]; then
  echo "[ERROR] install-server-linux.sh not found next to install-web.sh." >&2
  echo "[ERROR] Use the official web bundle or run scripts/install-server-linux.sh from the repository." >&2
  exit 1
fi

exec bash "$INSTALLER" "$@"
