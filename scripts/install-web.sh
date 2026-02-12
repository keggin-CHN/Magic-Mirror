#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTALL_DIR="${INSTALL_DIR:-$PROJECT_ROOT}"
WEB_PORT="${WEB_PORT:-8033}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
SERVICE_NAME="${SERVICE_NAME:-magic-mirror-web}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-root}}"
SKIP_WEB_BUILD="${SKIP_WEB_BUILD:-0}"

export DEBIAN_FRONTEND=noninteractive

echo "==> Installing system packages..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  python3 \
  python3-venv \
  python3-pip \
  ffmpeg \
  build-essential \
  rsync

if [ "$INSTALL_DIR" != "$PROJECT_ROOT" ]; then
  echo "==> Copying project to ${INSTALL_DIR}..."
  mkdir -p "$INSTALL_DIR"
  rsync -a --delete \
    --exclude node_modules \
    --exclude .git \
    --exclude dist \
    --exclude dist-web \
    --exclude data \
    "$PROJECT_ROOT"/ "$INSTALL_DIR"/
fi

cd "$INSTALL_DIR"

echo "==> Preparing data directories..."
mkdir -p "$INSTALL_DIR/data/web/uploads" "$INSTALL_DIR/data/web/library"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/data"

if [ "$SKIP_WEB_BUILD" != "1" ]; then
  if [ ! -f "dist-web/index.html" ]; then
    echo "==> Building web frontend..."
    if ! command -v node >/dev/null 2>&1; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
      apt-get install -y nodejs
    fi
    if ! command -v pnpm >/dev/null 2>&1; then
      if command -v corepack >/dev/null 2>&1; then
        corepack enable
        corepack prepare pnpm@8.5.1 --activate
      else
        npm install -g pnpm@8.5.1
      fi
    fi
    pnpm install --no-frozen-lockfile
    VITE_OUT_DIR=dist-web pnpm build
  else
    echo "==> dist-web exists; skip frontend build."
  fi
fi

echo "==> Setting up Python venv..."
python3 -m venv "$INSTALL_DIR/.venv-web"
source "$INSTALL_DIR/.venv-web/bin/activate"
pip install --upgrade pip
pip install -r src-python/requirements.txt
pip install -e src-python --no-deps
deactivate

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "==> Installing systemd service: ${SERVICE_NAME}"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=MagicMirror Web Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv-web/bin/python $INSTALL_DIR/src-python/web_server.py
Restart=on-failure
Environment=WEB_HOST=$WEB_HOST
Environment=WEB_PORT=$WEB_PORT
User=$SERVICE_USER
Group=$SERVICE_USER

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

echo "==> Service status:"
systemctl status "$SERVICE_NAME" --no-pager

echo "==> Done. Web server running on http://$WEB_HOST:$WEB_PORT"